"""Short, independent correctness probes and MTP counter checks (no trajectory prompts)."""
import argparse
import json
from pathlib import Path
import re
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = 'http://127.0.0.1:30002'


def get(path):
    with opener.open(base + path, timeout=120) as response:
        return response.read()


def counters(raw):
    result = {}
    for line in raw.decode().splitlines():
        if line.startswith('vllm:spec_decode_') and '_total{' in line:
            name = line.split('{', 1)[0]
            result[name] = result.get(name, 0) + float(line.rsplit(' ', 1)[1])
    return result


before = get('/metrics')
(args.output/'metrics-before.txt').write_bytes(before)
cases = [
    ('multiply', 'What is 17 multiplied by 19? Give only the integer as your final answer.', lambda x: x.strip() == '323'),
    ('sort', 'Sort these integers in ascending order: 12, -4, 7, 0, 7. Return only a JSON array as your final answer.', lambda x: json.loads(x) == [-4, 0, 7, 7, 12]),
    ('sequence', 'Return only a JSON array containing every integer from 1 through 40 inclusive, in order. Do not omit any number.', lambda x: json.loads(x) == list(range(1, 41))),
]
results = []
for name, prompt, check in cases:
    payload = {'model':'GLM-5.3', 'messages':[{'role':'user','content':prompt}],
               'reasoning_effort':'high', 'temperature':0, 'max_tokens':2048, 'stream':False}
    request = urllib.request.Request(base+'/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    start = time.monotonic()
    with opener.open(request, timeout=600) as response:
        body = json.load(response)
    elapsed = time.monotonic()-start
    answer = body['choices'][0]['message'].get('content') or ''
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', answer.strip())
    try:
        passed = bool(check(cleaned))
    except (ValueError, TypeError):
        passed = False
    item = {'case':name, 'passed':passed, 'elapsed_s':elapsed, 'request':payload, 'response':body}
    results.append(item)
    (args.output/(name+'.json')).write_text(json.dumps(item,indent=2)+'\n')
    print(json.dumps({'case':name,'passed':passed,'elapsed_s':elapsed,'answer':answer,'usage':body.get('usage')}),flush=True)
time.sleep(2)
after = get('/metrics')
(args.output/'metrics-after.txt').write_bytes(after)
a,b=counters(before),counters(after)
delta={k:v-a.get(k,0) for k,v in b.items()}
drafted=delta.get('vllm:spec_decode_num_draft_tokens_total',0)
accepted=delta.get('vllm:spec_decode_num_accepted_tokens_total',0)
summary={'passed':all(r['passed'] for r in results) and drafted>0 and accepted>0,
         'correctness_checks':len(results), 'counter_delta':delta,
         'acceptance_rate':accepted/drafted if drafted else None,
         'note':'Limited smoke checks; not proof of general output equivalence or SWE solving ability.'}
(args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
raise SystemExit(0 if summary['passed'] else 1)
