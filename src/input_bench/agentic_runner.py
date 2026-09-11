"""Independent frozen-session replay runs and offline reports using BenchmarkSender."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from input_bench import __version__
from input_bench.compiler import sha256_file
from input_bench.metrics import distribution
from input_bench.replay_bundle import validate_bundle, write_json
from input_bench.sender import BenchmarkCancelled, BenchmarkSender, SenderConfig
from input_bench.targets import TargetCapabilities, api_url

ALIASES = {'dsv4-flash': 'deepseek-v4-flash', 'glm5.3': 'ZHIPU/GLM-5.3',
           'glm5.3-flash': 'ZHIPU/GLM-5.3-Flash'}


def reasoning_parameters(profile, run):
    if profile == 'aliyun-chat':
        return {'enable_thinking': run['enable_thinking'], 'reasoning_effort': run['reasoning_effort']}
    if profile == 'sglang-dsv4-0731':
        # Installed SGLang's DeepSeek V4 parser uses explicit_thinking, not
        # Alibaba's top-level enable_thinking. The effort remains top-level.
        return {'chat_template_kwargs': {'thinking': run['enable_thinking']},
                'reasoning_effort': run['reasoning_effort']}
    if profile in {'sglang-glm53', 'sglang-glm53-flash', 'vllm-glm53'}:
        # Both GLM-5.3 checkpoint templates always open <think> and inject
        # Reasoning Effort from this kwarg. It has no enable_thinking toggle.
        if run['enable_thinking'] is not True:
            raise ValueError('GLM-5.3 templates require thinking enabled')
        return {'reasoning_effort': run['reasoning_effort']}
    raise ValueError('unsupported target_profile: ' + str(profile))


def comparable_capabilities(config):
    caps = dict(config['capabilities'])
    expected = reasoning_parameters(config.get('target_profile', 'aliyun-chat'), config['run'])
    if caps.get('extra_parameters') != expected:
        raise ValueError('saved reasoning parameters differ from declared target profile')
    # Compare logical controls separately; retain the actual wire mapping in
    # effective configs and report parameter differences explicitly.
    caps.pop('extra_parameters')
    return caps


def utc():
    return datetime.now(timezone.utc).isoformat()


def code_version():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {'version': __version__, 'commit': git('rev-parse', 'HEAD'),
            'dirty': bool(git('status', '--porcelain')),
            'implementation_sha256': {p.name: sha256_file(p) for p in [Path(__file__),
                Path(__file__).with_name('sender.py'), Path(__file__).with_name('targets.py'),
                Path(__file__).with_name('sse.py')]}}


def config_and_bundle(config_path, args):
    cfg = json.loads(config_path.read_text())
    selected = getattr(args, 'bundle', None) or os.getenv('INPUT_BENCH_WORKLOAD') or cfg['bundle']
    bundle = Path(selected).expanduser()
    if bundle.name == 'workload.jsonl': bundle = bundle.parent
    if not bundle.is_absolute(): bundle = config_path.resolve().parent / bundle
    requests, manifest, sessions = validate_bundle(bundle)
    run = dict(cfg['planned_run'])
    run['base_url'] = getattr(args, 'base_url', None) or os.getenv('ALIYUN_BASE_URL') or run['base_url']
    run['timeout_s'] = getattr(args, 'timeout', None) or run.get('timeout_s', 300)
    api_url(run['base_url'])
    if cfg['output_cap'] != manifest['compiler_parameters']['output_cap']:
        raise ValueError('config output_cap differs from frozen bundle')
    if run['warmup'] != 0 or run['repeats'] != 1 or run['retries'] != 0:
        raise ValueError('this initial replay supports warmup=0, repeats=1, retries=0 only')
    if run['session_concurrency'] < 1 or run['http_concurrency'] < 1 or run['timeout_s'] <= 0:
        raise ValueError('concurrency and timeout must be positive')
    if not run['stream'] or run['enable_thinking'] is not True or run['reasoning_effort'] not in {'high', 'max'}:
        raise ValueError('this three-model experiment requires streaming, thinking enabled and common effort high/max')
    chosen = getattr(args, 'model', 'all')
    models = run['models'] if chosen == 'all' else [ALIASES.get(chosen.lower(), chosen)]
    if not models or len(set(models)) != len(models) or any(m not in run['models'] for m in models):
        raise ValueError('model must be selected from configured models')
    profile = cfg.get('target_profile', 'aliyun-chat')
    target = TargetCapabilities(extra_parameters=reasoning_parameters(profile, run))
    if run.get('auth', 'env') not in {'env', 'none'}:
        raise ValueError('auth must be env or none')
    for model in models:
        for request in requests: target.payload(request, model, run['stream'])
    effective = {'schema_version': 'agentic-run-v1', 'created_at_utc': utc(),
        'workload_sha256': manifest['workload_sha256'], 'manifest_sha256': sha256_file(bundle/'manifest.json'),
        'manifest_content_sha256': manifest['content_sha256'],
        'session_ids': [s['session_id'] for s in sessions['selected']],
        'request_ids': [r.request_id for r in requests], 'planned_requests': len(requests),
        'backend': 'sglang' if profile.startswith('sglang-') else 'openai-compatible',
        'target_profile': profile, 'deployment': cfg.get('deployment'),
        'session_request_ids': {s['session_id']: s['request_ids'] for s in sessions['selected']},
        'reference_tokenizer': manifest['tokenizer'], 'output_cap': cfg['output_cap'],
        'normalization': manifest['normalization'], 'run': run, 'models': models,
        'capabilities': asdict(target), 'omitted_sampling': {k: sorted({str(r.sampling.get(k)) for r in requests})
                                                     for k in target.omitted_sampling},
        'code': code_version(), 'network': cfg.get('network', {'client_location': 'not_recorded'}),
        'pricing': cfg.get('pricing'), 'stop_on_error': not getattr(args, 'continue_on_error', False), 'reported_output_cap_tolerance': 10,
        'reasoning_verification': 'requested; provider internal effort is not observable',
        'client_recount': None, 'preflight': 'offline only; no health, models, metrics or warmup calls'}
    return requests, effective, target


def read_key(env_name, env_file=None):
    # Read only the named variable/file; never inspect shell history or unrelated credentials.
    value = os.environ.get(env_name, '').strip()
    if not value and env_file:
        path = Path(env_file)
        if path.stat().st_mode & 0o077:
            raise ValueError('Key file must be private; run chmod 600 on the specified file')
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith('export '): line = line[7:].strip()
            name, separator, raw = line.partition('=')
            if separator and name.strip() == env_name:
                value = raw.strip()
                if len(value) > 1 and value[0] == value[-1] and value[0] in '\"\'': value = value[1:-1]
    if not value:
        raise ValueError(f'{env_name} is unavailable; export it in this terminal or pass --env-file .env')
    if any(c.isspace() for c in value): raise ValueError('API key contains whitespace')
    return value


class Journal:
    """Stream checkpoints on one writer thread; drain/fsync outside sender measurement."""
    def __init__(self, path):
        self.queue = queue.SimpleQueue()
        self.error = None
        self.handle = path.open('x')
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            for row in iter(self.queue.get, None):
                self.handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                self.handle.flush()
            os.fsync(self.handle.fileno())
        except Exception as exc:
            self.error = type(exc).__name__
        finally:
            self.handle.close()

    def record(self, event):
        if self.error: raise RuntimeError('event journal failed: ' + self.error)
        self.queue.put(event.to_dict())

    def close(self):
        self.queue.put(None); self.thread.join()
        if self.error: raise RuntimeError('event journal failed: ' + self.error)


def summarize(rows, bounds, config):
    expected = config['request_ids']
    by_id = {e['request_id']: e for e in rows}
    complete_rows = [e for e in rows if e.get('request_end_ns') is not None and not e.get('error_type')]
    starts = [e['request_start_ns'] for e in rows if e.get('request_start_ns') is not None]
    ends = [e['request_end_ns'] for e in rows if e.get('request_end_ns') is not None]
    start, end = bounds.get('monotonic_start_ns'), bounds.get('monotonic_end_ns')
    total = (end-start)/1e9 if start is not None and end is not None else None
    delta = lambda e, key: ((e[key]-e['request_start_ns'])/1e6
        if e.get(key) is not None and e.get('request_start_ns') is not None else None)
    sessions = []
    for sid in config['session_ids']:
        own = [e for e in rows if e.get('session_id') == sid]
        a = [e['request_start_ns'] for e in own if e.get('request_start_ns') is not None]
        b = [e['request_end_ns'] for e in own if e.get('request_end_ns') is not None]
        sessions.append({'session_id': sid, 'session_duration_s': (max(b)-min(a))/1e9 if a and b else None,
                         'observed_requests': len(own),
                         'complete': set(e['request_id'] for e in own) == set(config['session_request_ids'][sid]) and all(not e.get('error_type') and e.get('request_end_ns') is not None for e in own),
                         'failed': sum(bool(e.get('error_type')) for e in own)})
    usable = (len(by_id) == len(rows) == len(expected) and set(by_id) == set(expected)
              and len(complete_rows) == len(expected) and total is not None and total > 0
              and all(start <= e['request_start_ns'] <= e['request_end_ns'] <= end for e in complete_rows))
    def usage_total(key):
        values = [(e.get('server_usage') or {}).get(key) for e in complete_rows]
        return sum(values) if values and all(type(v) is int and v >= 0 for v in values) else None
    errors = Counter(e.get('error_type') for e in rows if e.get('error_type'))
    answerless = sum(not e.get('answer_text') for e in complete_rows)
    outputs = usage_total('completion_tokens')
    return {'schema_version': 'agentic-summary-v1', 'complete': usable,
        'experiment_config': config, 'measurement': {**bounds, 'duration_s': total,
            'sender_total_s': total, 'request_span_s': (max(ends)-min(starts))/1e9 if starts and ends else None,
            'cumulative_request_e2e_s': sum(delta(e,'request_end_ns') or 0 for e in rows)/1000},
        'counts': {'planned':len(expected), 'started':len(starts), 'completed':len(complete_rows),
            'failed':sum(v for k,v in errors.items() if k not in {'cancelled','not_sent'}),
            'cancelled':errors['cancelled'], 'not_sent':errors['not_sent'],
            'missing':len(set(expected)-set(by_id)), 'answerless':answerless},
        'latency_ms': {'request_e2e':distribution([delta(e,'request_end_ns') for e in complete_rows]),
            'ttft_observed':distribution([delta(e,'first_content_ns') for e in complete_rows
                                         if e.get('streaming') and e.get('first_content_ns') is not None]),
            'answer_ttft':distribution([delta(e,'first_answer_ns') for e in complete_rows
                                       if e.get('streaming') and e.get('first_answer_ns') is not None]),
            'client_queue_delay':distribution([e['client_queue_delay_ns']/1e6 for e in rows
                                               if e.get('client_queue_delay_ns') is not None])},
        'tokens': {'reference_input': sum(e.get('reference_input_tokens') or 0 for e in rows),
                   'server_prompt':usage_total('prompt_tokens'), 'server_completion':outputs,
                   'unknown_usage_requests':sum(not e.get('server_usage') for e in complete_rows),
                   'source':'server_usage; reasoning may already be included, never add it again'},
        'output_tokens_per_s':outputs/total if outputs is not None and total else None,
        'finish_reasons':dict(Counter(e.get('finish_reason') or 'unknown' for e in rows)),
        'errors':dict(errors), 'sessions':sessions,
        'notes':['sender_total excludes compile, tokenizer loading, final artifact export and journal drain; '
                  'includes request parsing, scheduling, progress logging and checkpoint queue enqueue overhead.',
                 'Background checkpoint writes overlap the run; observe client CPU/IO for formal measurements.',
                 'Fixed recorded histories; high across models is not equal compute; no quality/resolved-rate evaluation.',
                 'No reasoning TPOT is calculated; provider hidden reasoning and token timing are not fully observable.']}


def persist(output, events, bounds, effective):
    rows = [e.to_dict() for e in events]
    write_json(output/'bounds.json', bounds)
    report = summarize(rows, bounds, effective)
    write_json(output/'summary.json', report)
    # Dict fields become JSON strings in Parquet, preserving variable provider usage schemas.
    import pyarrow as pa
    import pyarrow.parquet as pq
    export = [{k: (json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v)
               for k,v in row.items()} for row in rows]
    pq.write_table(pa.Table.from_pylist(export), output/'results.parquet')
    write_json(output/'state.json', {'status':'complete' if report['complete'] else 'incomplete',
                                   'updated_at_utc':utc(), 'counts':report['counts']})
    return report


async def run_models(requests, effective, target, output_root, run_id, key):
    # Import export dependencies before any paid request.
    import aiohttp
    import pyarrow.parquet
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id) or run_id in {'.','..'}:
        raise ValueError('run-id must be a simple directory name')
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    index = {'created_at_utc':utc(), 'models':effective['models'], 'results':[],
             'workload_sha256':effective['workload_sha256'], 'status':'running'}
    write_json(run_dir/'run-index.json',index)
    for model in effective['models']:
        label = model.replace('/','_')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',label): raise ValueError('invalid model directory label')
        output = run_dir / label
        output.mkdir()
        config = {**effective, 'model':model, 'run_id':run_id, 'repeat':1, 'created_at_utc':utc()}
        write_json(output/'effective-config.json',config)
        write_json(output/'bounds.json',{'planned_requests':len(requests), 'wall_start_utc':utc()})
        write_json(output/'state.json',{'status':'running', 'planned':len(requests)})
        journal = Journal(output/'events.jsonl')
        count = 0
        def progress(event):
            nonlocal count
            journal.record(event); count += 1
            if event.request_start_ns is not None:
                print(f'{model}: {count}/{len(requests)} finish={event.finish_reason} '
                      f'error={event.error_type} output_tokens={event.output_tokens}', flush=True)
        cfg = effective['run']
        sender = BenchmarkSender(SenderConfig(cfg['base_url'], model, stream=cfg['stream'], api_key=key,
            timeout_s=cfg['timeout_s'], max_concurrency=cfg['http_concurrency'],
            pool_size=cfg['http_concurrency'], response_max_chars=100000,
            backend='openai-compatible', target=target, session_concurrency=cfg['session_concurrency'],
            stop_on_error=effective['stop_on_error'], enforce_reported_output_cap=True), on_event=progress)
        cancelled = False
        try:
            events, bounds = await sender.run(requests)
        except BenchmarkCancelled as exc:
            events, bounds = exc.events, exc.bounds
            cancelled = True
        finally:
            journal.close()
        report = persist(output, events, bounds, config)
        index['results'].append({'model':model, 'path':str(output), 'complete':report['complete'],
                                 'sender_total_s':report['measurement']['sender_total_s']})
        index['updated_at_utc'] = utc()
        write_json(run_dir/'run-index.json',index)
        if cancelled or (not report['complete'] and effective['stop_on_error']): break
    index['status'] = ('complete' if len(index['results']) == len(index['models'])
                       and all(r['complete'] for r in index['results']) else 'incomplete')
    write_json(run_dir/'run-index.json',index)
    print('Results:',run_dir,flush=True)
    return 0 if index['status'] == 'complete' else 1


def rebuild(path):
    config = json.loads((path/'effective-config.json').read_text())
    bounds = json.loads((path/'bounds.json').read_text())
    rows = []
    with (path/'events.jsonl').open() as f:
        for line in f:
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: break  # crashed final partial line remains an incomplete run
    return summarize(rows,bounds,config)


def compare(paths, output):
    if len(paths) < 2: raise ValueError('compare requires at least two model result directories')
    summaries = [rebuild(p) for p in paths]
    first = summaries[0]['experiment_config']
    mismatches = []
    for s in summaries:
        cfg = s['experiment_config']
        for key in ['workload_sha256','manifest_content_sha256','session_ids','request_ids','output_cap','omitted_sampling']:
            if cfg[key] != first[key]: mismatches.append(key)
        if comparable_capabilities(cfg) != comparable_capabilities(first):
            mismatches.append('capabilities')
        for key in ['session_concurrency','http_concurrency','stream','reasoning_effort','enable_thinking','repeats','warmup','retries']:
            if cfg['run'][key] != first['run'][key]: mismatches.append(key)
        if not s['complete']: mismatches.append('incomplete_run')
    records=[]
    baseline=summaries[0]['measurement']['sender_total_s']
    for path, s in zip(paths,summaries):
        total=s['measurement']['sender_total_s']
        records.append({'model':s['experiment_config']['model'], 'result_directory':str(path),
            'complete':s['complete'], 'sender_total_s':total,
            'ratio_to_first':total/baseline if not mismatches and total is not None and baseline else None,
            'completed_requests':s['counts']['completed'], 'planned_requests':s['counts']['planned'],
            'answerless_requests':s['counts']['answerless'], 'server_output_tokens':s['tokens']['server_completion']})
    report={'comparable_as_endpoint_experience':not mismatches, 'mismatches':sorted(set(mismatches)),
        'wire_parameter_mappings_differ': any(s['experiment_config']['capabilities'] != first['capabilities'] for s in summaries),
        'ratio_definition':'this sender_total / first directory sender_total; >1 means slower',
        'limitations':['Different model, reasoning internals, cache, network and output lengths are not aligned.',
                       'GLM output cap semantics remain unverified. No inference about CVM/TEE overhead.'],
        'runs':records, 'effective_configs':[s['experiment_config'] for s in summaries]}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'comparison.json',report)
    with (output/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    lines=['# 固定轨迹 API 耗时对比','',f"可进行完整端点体验对比：{not mismatches}",
           '','| 模型 | sender_total_s | 相对首目录耗时 | 完成请求 |','|---|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['model']} | {r['sender_total_s']} | {r['ratio_to_first']} | {r['completed_requests']}/{r['planned_requests']} |")
    lines.extend(['','失败或缺失请求不计算有效耗时比。不同模型、输出量、思考实现、缓存与网络会影响结果；不归因于 CVM/TEE。',
                  '', 'Mismatch: '+', '.join(sorted(set(mismatches)))])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ['dry-run','run']:
        p=sub.add_parser(name)
        p.add_argument('--config',type=Path,required=True)
        p.add_argument('--model',default='dsv4-flash',help='dsv4-flash / glm5.3 / glm5.3-flash / all, or configured model ID')
        p.add_argument('--bundle',type=Path)
        p.add_argument('--base-url')
        p.add_argument('--timeout',type=float)
        p.add_argument('--continue-on-error',action='store_true',help='finish remaining replay after failed requests; run remains incomplete')
        if name=='run':
            p.add_argument('--env-file',type=Path,help='explicit private Key file; environment takes precedence')
            p.add_argument('--output-root',type=Path)
            p.add_argument('--run-id',default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    p=sub.add_parser('compare')
    p.add_argument('directories',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.action=='compare':
            report=compare(args.directories,args.output)
            print(json.dumps(report['runs'],indent=2)); return 0
        requests,effective,target=config_and_bundle(args.config,args)
        print(json.dumps({'models':effective['models'], 'requests_per_model':len(requests),
                          'total_requests':len(requests)*len(effective['models']),
                          'session_count':len(effective['session_ids']), 'workload_sha256':effective['workload_sha256'],
                          'run':effective['run'], 'omitted_sampling':effective['omitted_sampling'],
                          'network':effective['network']},ensure_ascii=False,indent=2))
        if args.action=='dry-run':
            print('Offline preflight passed; no key read, no API calls.'); return 0
        key=(None if effective['run'].get('auth') == 'none'
             else read_key(effective['run']['api_key_env'],args.env_file))
        cfg=json.loads(args.config.read_text())
        root=args.output_root or Path(os.getenv('INPUT_BENCH_RESULTS_ROOT') or cfg.get('results_root','../runs/agentic-replay/aliyun-high-20/results'))
        if not root.is_absolute(): root=args.config.resolve().parent/root
        return asyncio.run(run_models(requests,effective,target,root,args.run_id,key))
    except (ValueError,OSError,KeyError,RuntimeError,ImportError) as exc:
        print(f'agentic-replay: {exc}',file=sys.stderr); return 2
    except KeyboardInterrupt:
        print('Interrupted; inspect saved events and state before starting a new run.',file=sys.stderr); return 130
