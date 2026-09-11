import json
import shutil
from collections import UserDict

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from input_bench.adapters.swe_agent import NORMALIZATION, SWEAgentTrajectoriesAdapter
from input_bench.replay_bundle import prepare_bundle, validate_bundle
from input_bench.tokenizer import WhitespaceTokenizer, count_input


def trajectory():
    return [{'role': 'system', 'system_prompt': 'system rule'},
            {'role': 'user', 'text': 'recorded issue'},
            {'role': 'ai', 'text': 'recorded action', 'tool_calls': [
                {'id': 'one', 'function': {'name': 'shell', 'arguments': '{"command":"never execute"}'}}]},
            {'role': 'tool', 'name': 'shell', 'tool_call_id': 'one', 'content': 'recorded result'},
            {'role': 'user', 'text': 'user observation'},
            {'role': 'ai', 'text': 'CURRENT_GOLD_ANSWER'}]


def fixture(path, count=3):
    rows=[{'instance_id': 'same-issue', 'trajectory': trajectory() if i%2==0 else json.dumps(trajectory()),
           'generated_patch': 'SECRET_PATCH', 'eval_logs': 'SECRET_EVAL', 'model_name': 'recording-model'}
          for i in range(count)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_complete_sessions_normalized_deterministic_and_portable(tmp_path):
    src=tmp_path/'source.jsonl'
    fixture(src)
    a=SWEAgentTrajectoriesAdapter(src,revision='fixed-revision',normalization=NORMALIZATION)
    sessions=list(a.iter_sessions())
    assert len(sessions)==3 and len({s[0].session_id for s in sessions})==3
    assert all(len(s)==2 for s in sessions)
    prompt=json.dumps([m.to_dict() for m in sessions[0][1].messages])
    assert 'recorded action' in prompt and 'recorded result' in prompt and 'user observation' in prompt
    assert 'never execute' in prompt and 'CURRENT_GOLD_ANSWER' not in prompt
    assert 'SECRET_PATCH' not in prompt and 'SECRET_EVAL' not in prompt
    assert all(set(m.to_dict())=={'role','content'} for s in sessions for turn in s for m in turn.messages)
    kwargs=dict(revision='fixed-revision',seed=42,max_sessions=2,output_cap=4096)
    for name in ['first','second']:
        prepare_bundle(src,tmp_path/name,WhitespaceTokenizer(),'whitespace',**kwargs)
    rows,m,s=validate_bundle(tmp_path/'first')
    assert len(rows)==4 and len(s['selected'])==2
    assert m['content_sha256']==validate_bundle(tmp_path/'second')[1]['content_sha256']
    assert (tmp_path/'first/workload.jsonl').read_bytes()==(tmp_path/'second/workload.jsonl').read_bytes()
    assert rows[1].parent_request_id==rows[0].request_id
    assert all(r.max_output_tokens==4096 for r in rows)
    moved=tmp_path/'moved'; moved.mkdir()
    shutil.copy2(src,moved/src.name)
    assert next(SWEAgentTrajectoriesAdapter(moved/src.name,revision='fixed-revision').iter_samples()).session_id==sessions[0][0].session_id
    with pytest.raises(FileExistsError):
        prepare_bundle(src,tmp_path/'first',WhitespaceTokenizer(),'whitespace',**kwargs)
    (tmp_path/'first/workload.jsonl').write_text('corrupted')
    with pytest.raises(ValueError,match='hash mismatch'): validate_bundle(tmp_path/'first')


def test_whole_session_rejection_and_shortfall(tmp_path):
    src=tmp_path/'source.jsonl'; fixture(src,2)
    original=[json.loads(s) for s in src.read_text().splitlines()]
    # A bad late step must invalidate the earlier assistant turn too.
    original[0]['trajectory'].append(4)
    src.write_text(''.join(json.dumps(r)+'\n' for r in original))
    m=prepare_bundle(src,tmp_path/'valid',WhitespaceTokenizer(),'whitespace',revision='r',max_sessions=5)
    assert m['counts']['sessions']==1 and m['counts']['session_shortfall']==4
    assert m['counts']['requests']==2
    assert validate_bundle(tmp_path/'valid')[2]['excluded'][0]['reason']=='malformed_step'
    with pytest.raises(ValueError,match='no eligible sessions'):
        prepare_bundle(src,tmp_path/'too-long',WhitespaceTokenizer(),'whitespace',revision='r',max_input_tokens=1)
    empty=tmp_path/'empty.jsonl'; empty.write_text('')
    with pytest.raises(ValueError,match='no eligible sessions'):
        prepare_bundle(empty,tmp_path/'empty',WhitespaceTokenizer(),'whitespace',revision='r')


def test_seeded_parquet_candidates_keep_original_rows(tmp_path):
    src=tmp_path/'data'; src.mkdir()
    for shard in range(2):
        pq.write_table(pa.Table.from_pylist([{'instance_id':str(i),'trajectory':json.dumps(trajectory())}
                                           for i in range(10)]),src/f'{shard}.parquet',row_group_size=3)
    m=prepare_bundle(src,tmp_path/'bundle',WhitespaceTokenizer(),'whitespace',revision='r',
                     max_sessions=3,candidate_sessions=5,seed=7)
    assert m['sampling_frame']['total_source_sessions']==20
    assert m['sampling_frame']['outside_candidate_pool']==15
    assert m['counts']['sessions']==3
    rows,_,_=validate_bundle(tmp_path/'bundle')
    for r in rows:
        assert r.metadata['source_row'] in m['sampling_frame']['selected_source_rows'][r.metadata['source_file']]


@pytest.mark.parametrize('ids', [[1,2,3,4,5], UserDict(input_ids=[1,2,3,4,5],attention_mask=[1]*5),
                                 {'input_ids': [[1,2,3,4,5]]}])
def test_reference_token_count_not_batchencoding_field_count(ids):
    class Tokenizer:
        def apply_chat_template(self,*a,**kw): return ids
    assert count_input(Tokenizer(),prompt=None,messages=[{'role':'user','content':'hello'}])==5
