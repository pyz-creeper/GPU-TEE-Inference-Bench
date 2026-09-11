"""Offline comparison of a completed vLLM replay with the fixed SGLang baseline."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from input_bench.agentic_runner import compare, rebuild
import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument('--run-id', default='vllm-glm53-pp2-mtp-high-20-r2')
args = parser.parse_args()
root = ROOT/'runs/agentic-replay/vllm-glm53-pp2-mtp-high-20'
if (root/'launches'/args.run_id/'pause.json').exists():
    raise SystemExit('This run was paused. Its raw wall time and in-flight latency are not valid uninterrupted benchmark measurements. Preserve it as an interrupted/segmented run; use a fresh run ID for an uninterrupted comparison.')
result = root/'results'/args.run_id/'GLM-5.3'
baseline = ROOT/'runs/agentic-replay/sglang-glm53-pp2-high-20/results/sglang-glm53-pp2-high-20-r1/GLM-5.3'
summary = rebuild(result)
assert summary == json.loads((result/'summary.json').read_text())
assert summary['complete'] and summary['counts']['completed'] == 376
rows = [json.loads(s) for s in (result/'events.jsonl').read_text().splitlines()]
assert [r['request_id'] for r in rows] == summary['experiment_config']['request_ids']
assert len({r['request_id'] for r in rows}) == 376
assert all(r['http_status'] == 200 and not r['error_type'] for r in rows)
assert all(not row['response_truncated'] for row in rows)
assert all(a['request_end_ns'] <= b['request_start_ns'] for a,b in zip(rows, rows[1:]))
assert len(summary['sessions']) == 20 and all(s['complete'] for s in summary['sessions'])
assert pq.read_metadata(result/'results.parquet').num_rows == 376
comparison_dir = root/'comparisons'/args.run_id
if comparison_dir.exists():
    # Re-reporting must revalidate the stored comparison without deleting it.
    with TemporaryDirectory(prefix='glm53-compare-') as temporary:
        comparison = compare([baseline, result], Path(temporary)/'comparison')
    assert comparison == json.loads((comparison_dir/'comparison.json').read_text())
else:
    comparison = compare([baseline, result], comparison_dir)
assert comparison['comparable_as_endpoint_experience']
old = rebuild(baseline)


def repeated_lines(events):
    """Diagnostic candidates only; not a correctness or SWE success metric."""
    candidates = []
    for event in events:
        lines = Counter(line.strip() for line in event['response_text'].splitlines()
                        if len(line.strip()) >= 5 and not line.strip().startswith('```'))
        if lines:
            line, count = lines.most_common(1)[0]
            if count >= 20:
                candidates.append({'request_id':event['request_id'],
                                   'line':line[:160], 'count':count})
    return candidates


repetitions = repeated_lines(rows)
reasoning_details = [(row.get('server_usage') or {}).get('completion_tokens_details') or {}
                     for row in rows]
reasoning_tokens = (sum(detail['reasoning_tokens'] for detail in reasoning_details)
                    if all('reasoning_tokens' in detail for detail in reasoning_details) else None)
baseline_rows = [json.loads(s) for s in (baseline/'events.jsonl').read_text().splitlines()]
baseline_repetitions = repeated_lines(baseline_rows)
old_sessions = {session['session_id']:session for session in old['sessions']}
session_comparison = []
for number, session in enumerate(summary['sessions'], 1):
    session_id = session['session_id']
    previous = old_sessions[session_id]
    session_comparison.append({'number':number, 'session_id':session_id,
        'requests':session['observed_requests'],
        'vllm_minutes':session['session_duration_s']/60,
        'sglang_minutes':previous['session_duration_s']/60,
        'vllm_output_tokens':sum(row['output_tokens'] or 0 for row in rows
                                if row['session_id'] == session_id),
        'sglang_output_tokens':sum(row['output_tokens'] or 0 for row in baseline_rows
                                  if row['session_id'] == session_id)})


def counters(path):
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith('vllm:spec_decode_') and '_total{' in line:
            name = line.split('{',1)[0]
            values[name] = values.get(name,0) + float(line.rsplit(' ',1)[1])
    return values


launch = root/'launches'/args.run_id
server_info = json.loads((launch/'server-before.json').read_text())
baseline_info = json.loads((ROOT/'runs/agentic-replay/sglang-glm53-pp2-high-20/deployment/server-ready.json').read_text())
sampling_seeds = {'vllm':server_info['vllm_config']['model_config']['seed'],
                  'sglang':baseline_info['random_seed'], 'request_seed':'omitted'}
before,after = counters(launch/'metrics-before.txt'),counters(launch/'metrics-after.txt')
delta = {k:v-before.get(k,0) for k,v in after.items()}
drafted = delta.get('vllm:spec_decode_num_draft_tokens_total',0)
accepted = delta.get('vllm:spec_decode_num_accepted_tokens_total',0)
assert drafted > 0 and 0 < accepted <= drafted
seconds = summary['measurement']['sender_total_s']
old_seconds = old['measurement']['sender_total_s']
report = {'run_id':args.run_id,'complete':True,'requests':376,'sessions':20,
          'vllm_seconds':seconds,'vllm_minutes':seconds/60,
          'baseline_seconds':old_seconds,'baseline_minutes':old_seconds/60,
          'faster_than_120_minutes':seconds < 7200,
          'time_reduction_percent':100*(1-seconds/old_seconds),
          'time_change_percent':100*(seconds/old_seconds-1),
          'speedup_vs_sglang':old_seconds/seconds,
          'mtp_counter_delta':delta,'mtp_acceptance_rate':accepted/drafted,
          'vllm_completion_tokens':summary['tokens']['server_completion'],
          'vllm_prompt_tokens':summary['tokens']['server_prompt'],
          'baseline_prompt_tokens':old['tokens']['server_prompt'],
          'reference_input_token_mismatches':sum(row['input_tokens'] != row['reference_input_tokens'] for row in rows),
          'vllm_reported_reasoning_tokens':reasoning_tokens,
          'baseline_completion_tokens':old['tokens']['server_completion'],
          'vllm_end_to_end_tokens_per_s':summary['tokens']['server_completion']/seconds,
          'baseline_end_to_end_tokens_per_s':old['tokens']['server_completion']/old_seconds,
          'vllm_length':summary['finish_reasons'].get('length',0),
          'vllm_answerless':summary['counts']['answerless'],
          'repeated_line_candidates':repetitions,
          'baseline_repeated_line_candidates':baseline_repetitions,
          'session_comparison':session_comparison,
          'validation':'exact request order, serial boundaries, 20 sessions, Parquet, offline rebuild and comparison passed',
          'sampling_seeds':sampling_seeds,
          'limitation':'Framework, kernels, KV dtype, server sampling seed and generated output length differ; a single replay does not isolate MTP speedup.'}
(root/(args.run_id+'-summary.json')).write_text(json.dumps(report,indent=2)+'\n')
lines=['# GLM-5.3 vLLM PP2 + MTP 实验结果','',
       f'本次 **{seconds/60:.2f} 分钟**，SGLang PP2 无 MTP 基线 **{old_seconds/60:.2f} 分钟**。',
       f'是否低于 120 分钟：**{"是" if seconds < 7200 else "否"}**。耗时变化：{report["time_change_percent"]:+.2f}%。','',
       '20 条会话、376/376 请求完整回放，均 HTTP 200；顺序、串行边界、Parquet 和离线重建通过。',
       f'MTP 接受率：{accepted/drafted:.2%}（{accepted:.0f}/{drafted:.0f} 草稿 tokens）。',
       f'输出 tokens：本次 {report["vllm_completion_tokens"]}，基线 {report["baseline_completion_tokens"]}。',
       f'vLLM 服务端报告的思考 tokens：{reasoning_tokens}（包含在上述输出总数中）。',
       f'输出总 tokens / 完整回放秒数：vLLM {report["vllm_end_to_end_tokens_per_s"]:.2f}，SGLang {report["baseline_end_to_end_tokens_per_s"]:.2f} tokens/s（包含 prefill 等开销，并非纯解码速度）。',
       f'达到输出上限 {report["vllm_length"]} 次，无最终答案 {report["vllm_answerless"]} 次。','',
       f'重复行诊断候选：本次 {len(repetitions)}、基线 {len(baseline_repetitions)} 个请求（同一非空行重复至少 20 次；不等于解题正确性判断）。','',
       '这是固定请求性能回放，不执行真实 agent 工具，不衡量 SWE 解题率。',
       '框架、内核、KV 精度和输出长度均有变化，结果不能单独归因于 MTP。',
       f'服务端采样 seed：vLLM {sampling_seeds["vllm"]}，SGLang {sampling_seeds["sglang"]}；请求未发送 seed。本轮为单次回放，生成文本并不要求相同。',
       '启动与独立 smoke 不计入正式计时；正式前后 MTP counters 差值只覆盖本次回放。','',
       f'[离线比较详情](comparisons/{args.run_id}/REPORT.md)',
       '[部署及运行命令](../../../docs/GLM53_VLLM_PP2_MTP_RUNBOOK.md)']
lines += ['', '## 逐会话比较', '',
          '| 会话序号 | 请求数 | vLLM 分钟 | SGLang 分钟 | vLLM 输出 tokens | SGLang 输出 tokens |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
for session in session_comparison:
    lines.append(f'| {session["number"]} | {session["requests"]} | {session["vllm_minutes"]:.2f} | '
                 f'{session["sglang_minutes"]:.2f} | {session["vllm_output_tokens"]} | {session["sglang_output_tokens"]} |')
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report,indent=2),flush=True)
