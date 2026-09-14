import asyncio
import json
import shutil
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp import web

from input_bench.agentic_runner import config_and_bundle, run_models, compare, rebuild, read_key, main
from input_bench.replay_bundle import prepare_bundle
from input_bench.schema import WorkloadRequest
from input_bench.sender import BenchmarkSender, SenderConfig, BenchmarkCancelled
from input_bench.targets import TargetCapabilities, api_url
from input_bench.tokenizer import WhitespaceTokenizer


async def server(handler):
    app = web.Application()
    app.router.add_post('/proxy/compatible-mode/v1/chat/completions', handler)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
    return runner, f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/proxy/compatible-mode/v1'


def request(i=0, sid='one', parent=None):
    return WorkloadRequest(str(i),i,'fixture','agent_coding','chat',5,4096,
        messages=[{'role':'user','content':'issue'}],session_id=sid,parent_request_id=parent)


def config(tmp_path):
    source = tmp_path/'data.jsonl'
    source.write_text(''.join(json.dumps({'instance_id':str(i), 'trajectory':[
        {'role':'user','content':f'issue {i}'}, {'role':'ai','content':'RECORDED_ACTION'},
        {'role':'user','content':'RECORDED_OBSERVATION'}, {'role':'ai','content':'GOLD'}]})+'\n' for i in range(3)))
    prepare_bundle(source,tmp_path/'bundle',WhitespaceTokenizer(),'whitespace',revision='r',max_sessions=3,output_cap=4096)
    cfg = {'bundle':'bundle', 'output_cap':4096, 'planned_run':{
        'models':['mock-a','mock-b'], 'base_url':'http://localhost', 'api_key_env':'MOCK_KEY',
        'stream':True, 'enable_thinking':True, 'reasoning_effort':'high',
        'session_concurrency':2, 'http_concurrency':1, 'repeats':1,'warmup':0,'retries':0}}
    path=tmp_path/'config.json'; path.write_text(json.dumps(cfg)); return path


def sse(choices, usage=None):
    return ('data: '+json.dumps({'choices':choices,'usage':usage},ensure_ascii=False)+'\n\n').encode()


async def test_two_endpoints_independent_replay_and_offline_compare(tmp_path):
    path=config(tmp_path)
    recorded={'mock-a':[],'mock-b':[]}
    active=0; peak=0
    async def handler(req):
        nonlocal active,peak
        body=await req.json()
        assert req.headers['Authorization']=='Bearer fake-key'
        assert set(body)=={'model','messages','stream','stream_options','max_tokens','enable_thinking','reasoning_effort'}
        assert body['enable_thinking'] is True and body['reasoning_effort']=='high'
        assert body['max_tokens']==4096
        recorded[body['model']].append(body['messages'])
        active+=1; peak=max(peak,active)
        await asyncio.sleep(.005)
        response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
        await response.prepare(req)
        data=sse([{'delta':{'role':'assistant'}}])
        data+=sse([{'delta':{'reasoning_content':'思考'}}])
        data+=sse([{'delta':{'content':'NEW_ANSWER_NEVER_REPLAY'},'finish_reason':'stop'}])
        data+=sse([],{'prompt_tokens':12,'completion_tokens':7,'completion_tokens_details':{'reasoning_tokens':5}})
        data+=b'data: [DONE]\n\n'
        # Split within the first Chinese character, exercising incremental UTF-8.
        cut=data.index('思'.encode())+1
        await response.write(data[:cut]); await response.write(data[cut:]); await response.write_eof()
        active-=1
        return response
    runner_a,url_a=await server(handler)
    runner_b,url_b=await server(handler)
    try:
        for model,url in [('mock-a',url_a),('mock-b',url_b)]:
            args=Namespace(bundle=None,base_url=url,timeout=3,model=model,continue_on_error=False)
            rows,effective,target=config_and_bundle(path,args)
            assert await run_models(rows,effective,target,tmp_path/'results',model,'fake-key')==0
            result=rebuild(tmp_path/'results'/model/model)
            assert result['complete'] and result['counts']['completed']==6
            assert result['tokens']['server_completion']==42  # reasoning not added twice
            events=[json.loads(line) for line in (tmp_path/'results'/model/model/'events.jsonl').read_text().splitlines()]
            byid={e['request_id']:e for e in events}
            for row in rows:
                if row.parent_request_id:
                    assert byid[row.parent_request_id]['request_end_ns']<=byid[row.request_id]['request_start_ns']
            assert all(e['reference_input_tokens']==r.input_tokens for e,r in zip(sorted(events,key=lambda e:e['sequence_no']),rows))
            assert all(e['client_output_tokens'] is None and e['output_token_source']=='server_usage' for e in events)
            assert 'fake-key' not in (tmp_path/'results'/model/model/'effective-config.json').read_text()
            assert (tmp_path/'results'/model/model/'results.parquet').is_file()
    finally:
        await runner_a.cleanup(); await runner_b.cleanup()
    # Both servers are gone; comparison must work entirely offline.
    assert recorded['mock-a']==recorded['mock-b']
    assert all('NEW_ANSWER_NEVER_REPLAY' not in json.dumps(m) for m in recorded['mock-a'])
    assert any('RECORDED_ACTION' in json.dumps(m) and 'RECORDED_OBSERVATION' in json.dumps(m) for m in recorded['mock-a'])
    assert peak==1
    report=compare([tmp_path/'results/mock-a/mock-a',tmp_path/'results/mock-b/mock-b'],tmp_path/'comparison')
    assert report['comparable_as_endpoint_experience']
    assert report['runs'][1]['ratio_to_first']>0
    # Missing requests cannot win a comparison with shorter elapsed time.
    shutil.copytree(tmp_path/'results/mock-b/mock-b',tmp_path/'broken-model')
    eventsfile=tmp_path/'broken-model/events.jsonl'
    eventsfile.write_text(eventsfile.read_text().splitlines()[0]+'\n')
    report=compare([tmp_path/'results/mock-a/mock-a',tmp_path/'broken-model'],tmp_path/'incomplete-comparison')
    assert not report['comparable_as_endpoint_experience']
    assert all(r['ratio_to_first'] is None for r in report['runs'])


async def test_active_sessions_bounded_and_cumulative_time_not_makespan():
    # Barrier ensures two sessions overlap, without fragile timing assertions.
    barrier=asyncio.Event(); arrived=0
    async def handler(req):
        nonlocal arrived
        arrived+=1
        if arrived==2: barrier.set()
        await asyncio.wait_for(barrier.wait(),3)
        await asyncio.sleep(.02)
        return web.Response(body=sse([{'delta':{'content':'ok'},'finish_reason':'stop'}])+b'data: [DONE]\n\n',content_type='text/event-stream')
    runner,url=await server(handler)
    rows=[request(0,'a'),request(1,'a','0'),request(2,'b'),request(3,'b','2'),request(4,'c'),request(5,'c','4')]
    try:
        events,bounds=await BenchmarkSender(SenderConfig(url,'m',target=TargetCapabilities(),
            session_concurrency=2,max_concurrency=2)).run(rows)
    finally: await runner.cleanup()
    sessions={}
    for e in events: sessions.setdefault(e.session_id,[]).append(e)
    a=sessions['a']; b=sessions['b']; c=sessions['c']
    assert c[0].request_start_ns>=min(a[-1].request_end_ns,b[-1].request_end_ns)
    cumulative=sum((e.request_end_ns-e.request_start_ns) for e in events)
    assert cumulative>bounds['monotonic_end_ns']-bounds['monotonic_start_ns']
    assert all(e.output_tokens is None for e in events)


@pytest.mark.parametrize('profile', ['sglang-dsv4-0731', 'sglang-glm53-flash', 'sglang-glm53', 'vllm-glm53'])
async def test_sglang_reasoning_mapping_no_auth_and_cloud_comparison(tmp_path, monkeypatch, profile):
    path = config(tmp_path)
    cfg = json.loads(path.read_text())
    requests, cloud, cloud_target = config_and_bundle(path, Namespace(model='mock-a'))
    cfg['target_profile'] = profile
    cfg['planned_run']['auth'] = 'none'
    cfg['planned_run'].pop('api_key_env')
    path.write_text(json.dumps(cfg))
    requests, local, local_target = config_and_bundle(path, Namespace(model='mock-b'))
    body = local_target.payload(requests[0], 'mock-b', True)
    assert 'enable_thinking' not in body
    if profile == 'sglang-dsv4-0731':
        assert body['chat_template_kwargs'] == {'thinking': True}
    else:
        assert 'chat_template_kwargs' not in body  # GLM template always thinks
    assert body['reasoning_effort'] == 'high'
    assert body['messages'] == cloud_target.payload(requests[0], 'mock-a', True)['messages']

    async def handler(req):
        data = await req.json()
        if data['model'] == 'mock-b':
            assert 'Authorization' not in req.headers
            assert data.get('chat_template_kwargs') == body.get('chat_template_kwargs')
        return web.Response(body=sse([{'delta':{'reasoning_content':'think'}}]) +
            sse([{'delta':{'content':'ok'},'finish_reason':'stop'}]) + b'data: [DONE]\n\n',
            content_type='text/event-stream')
    runner, url = await server(handler)
    try:
        for effective, target, key in [(cloud, cloud_target, 'fake-key'), (local, local_target, None)]:
            effective['run']['base_url'] = url
            model = effective['models'][0]
            assert await run_models(requests, effective, target, tmp_path/'results', model, key) == 0
    finally:
        await runner.cleanup()
    report = compare([tmp_path/'results/mock-a/mock-a', tmp_path/'results/mock-b/mock-b'], tmp_path/'comparison')
    assert report['comparable_as_endpoint_experience']
    assert report['wire_parameter_mappings_differ']
    saved_path = tmp_path/'results/mock-b/mock-b/effective-config.json'
    saved = json.loads(saved_path.read_text())
    saved['capabilities']['extra_parameters']['reasoning_effort'] = 'low'
    saved_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='reasoning parameters differ'):
        compare([tmp_path/'results/mock-a/mock-a', tmp_path/'results/mock-b/mock-b'], tmp_path/'bad-comparison')
    # Explicit none-auth also works through CLI without inspecting any Key.
    def forbidden(*args):
        raise AssertionError('read_key must not be called')
    monkeypatch.setattr('input_bench.agentic_runner.read_key', forbidden)
    async def fake_run(*args):
        assert args[-1] is None
        return 0
    monkeypatch.setattr('input_bench.agentic_runner.run_models', fake_run)
    assert await asyncio.to_thread(main, ['run', '--config', str(path), '--model', 'mock-b']) == 0


@pytest.mark.parametrize('status',[401,429])
async def test_errors_stop_and_redact(tmp_path,status):
    async def handler(req): return web.Response(status=status,text='fake-key')
    runner,url=await server(handler)
    try:
        rows,effective,target=config_and_bundle(config(tmp_path),Namespace(model='mock-a',base_url=url,timeout=3,continue_on_error=False,bundle=None))
        effective['run']['session_concurrency']=1  # the initial paid-run profile
        assert await run_models(rows,effective,target,tmp_path/'results','failure','fake-key')==1
    finally: await runner.cleanup()
    report=rebuild(tmp_path/'results/failure/mock-a')
    assert not report['complete'] and report['counts']['not_sent']==5
    assert report['counts']['failed']==1
    assert 'fake-key' not in (tmp_path/'results/failure/mock-a/events.jsonl').read_text()


async def test_failed_parent_releases_child_when_continuing():
    calls=0
    async def handler(req):
        nonlocal calls
        calls+=1
        if calls==1: return web.Response(status=500)
        return web.Response(body=sse([{'delta':{'content':'ok'},'finish_reason':'stop'}])+b'data: [DONE]\n\n',content_type='text/event-stream')
    runner,url=await server(handler)
    try:
        events,_=await BenchmarkSender(SenderConfig(url,'m',target=TargetCapabilities(),session_concurrency=1)).run([request(),request(1,parent='0')])
    finally: await runner.cleanup()
    assert calls==2 and events[0].error_type=='http_error' and events[1].error_type is None


async def test_cancellation_preserves_planned_counts():
    began=asyncio.Event()
    release=asyncio.Event()
    async def handler(req):
        began.set(); await release.wait(); return web.Response(status=200)
    runner,url=await server(handler)
    sender=BenchmarkSender(SenderConfig(url,'m',target=TargetCapabilities(),session_concurrency=1))
    task=asyncio.create_task(sender.run([request(),request(1,parent='0')]))
    await began.wait(); task.cancel()
    try:
        with pytest.raises(BenchmarkCancelled) as exc: await task
        assert len(exc.value.events)==2 and all(e.error_type=='cancelled' for e in exc.value.events)
    finally:
        release.set(); await runner.cleanup()


@pytest.mark.parametrize('base,expected',[
    ('http://host','http://host/v1/chat/completions'),
    ('http://host/v1/','http://host/v1/chat/completions'),
    ('http://host/proxy/compatible-mode/v1','http://host/proxy/compatible-mode/v1/chat/completions'),
    ('http://host/proxy','http://host/proxy/v1/chat/completions')])
def test_url_prefix(base,expected): assert api_url(base)==expected


def test_payload_protection_and_key_file(tmp_path,monkeypatch):
    monkeypatch.delenv('MOCK_KEY',raising=False)
    for field in ['messages','max_tokens','max_completion_tokens','tools','request_id','stream']:
        with pytest.raises(ValueError): TargetCapabilities(extra_parameters={field:1}).payload(request(),'m',True)
    row=request(); row.sampling['ignore_eos']=True
    with pytest.raises(ValueError): TargetCapabilities().payload(row,'m',True)
    p=tmp_path/'.env'; p.write_text('DONT_READ=other\nMOCK_KEY="fake-key"\n'); p.chmod(0o600)
    assert read_key('MOCK_KEY',p)=='fake-key'
    monkeypatch.setenv('MOCK_KEY','env-key')
    assert read_key('MOCK_KEY',p)=='env-key'
    monkeypatch.delenv('MOCK_KEY')
    p.chmod(0o644)
    with pytest.raises(ValueError): read_key('MOCK_KEY',p)


def test_dry_run_never_reads_key(tmp_path,monkeypatch):
    def fail(*a,**kw): raise AssertionError('dry-run must not read secrets')
    monkeypatch.setattr('input_bench.agentic_runner.read_key',fail)
    assert main(['dry-run','--config',str(config(tmp_path)),'--model','all'])==0


@pytest.mark.parametrize('kind,expected',[
    ('tool','unexpected_tool_call'),('cutoff','parse_error'),('badjson','parse_error'),
    ('length',None),('excess','reported_output_exceeds_cap')])
async def test_generic_response_outcomes(kind,expected):
    async def handler(req):
        if kind=='badjson': data=b'data: {bad}\n\n'
        elif kind=='tool': data=sse([{'delta':{'tool_calls':[{'id':'x'}]},'finish_reason':'tool_calls'}])
        else: data=sse([{'delta':{'reasoning_content':'思考'},'finish_reason':'length' if kind=='length' else 'stop'}])
        if kind=='excess': data+=sse([],{'completion_tokens':5000})
        if kind!='cutoff': data+=b'data: [DONE]\n\n'
        return web.Response(body=data,content_type='text/event-stream')
    runner,url=await server(handler)
    try:
        events,_=await BenchmarkSender(SenderConfig(url,'m',target=TargetCapabilities(),
            enforce_reported_output_cap=True)).run([request()])
    finally: await runner.cleanup()
    assert events[0].error_type==expected
    if kind=='length':
        assert events[0].reasoning_text=='思考' and not events[0].answer_text
        assert events[0].first_answer_ns is None and events[0].output_tokens is None
