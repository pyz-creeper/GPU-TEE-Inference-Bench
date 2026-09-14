#!/usr/bin/env python3
"""Prepare/validate a frozen trajectory bundle and report scale offline. Never calls APIs."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from statistics import median
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from input_bench.compiler import sha256_file
from input_bench.replay_bundle import prepare_bundle, validate_bundle, write_json
from input_bench.tokenizer import load_tokenizer


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'scenarios/aliyun_swe_trajectory_high.json')
    p.add_argument('--bundle', type=Path, help='override output; existing bundles are validated, never overwritten')
    args = p.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    def path(value):
        item = Path(value).expanduser()
        return item if item.is_absolute() else args.config.resolve().parent / item
    bundle = args.bundle.resolve() if args.bundle else path(cfg['bundle']).resolve()
    opts = {k: cfg[k] for k in ['seed','max_sessions','candidate_sessions','max_input_tokens','output_cap']}
    if bundle.exists():
        requests, manifest, sessions = validate_bundle(bundle)
        if manifest['compiler_parameters'] != opts or manifest['dataset_revision'] != cfg['dataset_revision']:
            raise ValueError('existing bundle uses different compile settings; choose a new --bundle')
        if manifest['tokenizer']['id'] != str(path(cfg['reference_tokenizer'])):
            raise ValueError('existing bundle has a different reference tokenizer')
    else:
        print('Preparing complete trajectories; no API requests.', flush=True)
        tokenizer_path = str(path(cfg['reference_tokenizer']))
        prepare_bundle(path(cfg['source']), bundle, load_tokenizer(tokenizer_path), tokenizer_path,
                       revision=cfg['dataset_revision'], **opts)
        requests, manifest, sessions = validate_bundle(bundle)
    total_input = sum(r.input_tokens for r in requests)
    grouped = {}
    for request in requests: grouped.setdefault(request.session_id, []).append(request)
    records = []
    for index, session in enumerate(sessions['selected'], 1):
        rows = grouped[session['session_id']]
        records.append({'index': index, 'session_id': session['session_id'], 'instance_id': session['instance_id'],
                        'source_file': session['source_file'], 'source_row': session['source_row'],
                        'recording_model': session['model_name'], 'requests': len(rows),
                        'input_tokens_sum': sum(r.input_tokens for r in rows),
                        'input_tokens_max': max(r.input_tokens for r in rows),
                        'recorded_output_tokens_sum': sum(r.metadata['reference_output_tokens'] for r in rows)})
    estimates = []
    for model, prices in cfg['pricing']['models'].items():
        for output_per_request in [1024,4096]:
            estimates.append({'model':model, 'assumed_average_billed_output_tokens':output_per_request,
                'input_cny_reference': total_input*prices['input']/1e6,
                'output_cny_scenario': len(requests)*output_per_request*prices['output']/1e6,
                'total_cny_scenario': (total_input*prices['input']+len(requests)*output_per_request*prices['output'])/1e6})
    report = {'state':'offline-scale-only', 'bundle':str(bundle), 'workload_sha256':manifest['workload_sha256'],
              'manifest_sha256':sha256_file(bundle/'manifest.json'), 'counts':manifest['counts'],
              'total_api_requests_for_all_models':len(requests)*len(cfg['planned_run']['models']),
              'total_reference_input_tokens_per_model':total_input, 'tokens':manifest['token_statistics'],
              'planned_run':cfg['planned_run'], 'pricing':cfg['pricing'], 'cost_scenarios':estimates,
              'sessions':records,
              'limitations':['Fixed recorded history, not live SWE-bench task solving.',
                'Same effort name is not equal compute across models.',
                'GLM direct-provider output-cap/reasoning semantics need live verification.',
                'No API requests were sent; cost scenarios are not billing guarantees.']}
    out = bundle.parent
    write_json(out/'scale.json',report)
    with (out/'sessions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    lines = ['# 阿里云 SWE-agent 固定轨迹：规模提案', '',
        '**仅离线准备，未调用 API。当前阶段先确认规模，再决定充值与执行。**','',
        f"- 完整 session：{len(records)}；独立 issue：{len({r['instance_id'] for r in records})}。",
        f"- 每模型请求：{len(requests)}；三个模型合计：{report['total_api_requests_for_all_models']}。",
        f"- 每模型累计 reference 输入：{total_input:,} tokens（历史前缀在每一轮重复计入）。",
        f"- 轮数 min/median/max：{min(r['requests'] for r in records)}/"
        f"{median(r['requests'] for r in records)}/{max(r['requests'] for r in records)}。",
        '- GLM-5.3 本地 tokenizer 作为 reference；server tokenizer 与计费可能不同。',
        f"- 从全部 {manifest['sampling_frame']['total_source_sessions']:,} 条源记录按 seed={cfg['seed']} 抽取 {manifest['sampling_frame']['candidate_sessions']} 条候选，再整条过滤、抽取{len(records)}条；不是全体合格轨迹的穷举统计。",
        f"- 输入限制 {cfg['max_input_tokens']:,} reference tokens；每轮请求 max_tokens={cfg['output_cap']:,}。",
        '- 三个模型均 enable_thinking=true / reasoning_effort=high；各跑一次，session/HTTP 并发1，warmup0，retries0。',
        '- 固定录制历史、不执行工具、不评价任务解决率；返回内容不影响下一轮输入。',
        '- GLM 直供接口的 max_tokens 是否包含 reasoning 尚未验证；下表不能当作费用上限。', '',
        '## 费用情景（元）', '',
        '按每百万 token 单价和 reference 输入计数计算；不计 cache、免费额度或 Flash 临时折扣。',
        '输出指实际计费的 reasoning + answer；1024/4096 是平均输出情景，非预测或承诺。','',
        '| 模型 | 每轮平均输出1024 | 每轮平均输出4096 |',
        '|---|---:|---:|']
    for model in cfg['planned_run']['models']:
        vals=[v['total_cny_scenario'] for v in estimates if v['model']==model]
        lines.append('| '+model+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    totals=[sum(v['total_cny_scenario'] for v in estimates if v['assumed_average_billed_output_tokens']==n) for n in [1024,4096]]
    lines.extend(['| 三模型合计 | '+' | '.join(f'{v:.2f}' for v in totals)+' |','',
        '价格来源（2026-09-07）：https://help.aliyun.com/zh/model-studio/model-pricing','',
        '## 冻结文件','',f"workload SHA-256: `{manifest['workload_sha256']}`",'',
        '`bundle/workload.jsonl`、`bundle/manifest.json`、`bundle/sessions.json`；逐 session 明细见 `sessions.csv`。'])
    (out/'SCALE_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['state','counts','total_api_requests_for_all_models',
        'total_reference_input_tokens_per_model','tokens']},ensure_ascii=False,indent=2))
    print('Report:',out/'SCALE_REPORT.md')
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f'prepare-agentic-replay: {exc}',file=sys.stderr)
        raise SystemExit(2)
