#!/usr/bin/env python3
"""Export local, anonymous raw-content evidence without credentials or host metadata.

This creates a local candidate, never uploads it. Model-generated task content is
retained. RL extraction-to-dispatch ordering is not invented where IDs are absent.
"""
import collections
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'submission_packages/mechanism_evidence'
FORBIDDEN=re.compile(r'wangwenbo|wenbwang|Wenbo Wang|Di Sang|3120265429|nebula-1999|/Users/|@bit\.edu|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|PRIVATE KEY',re.I)

def write(name, rows):
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    match=FORBIDDEN.search(text)
    assert not match, f'Sensitive/identity marker in {name}; export stopped'
    dest=OUT/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(text)

def main():
    summary={}
    # All tasks, no outcome-conditioned sampling. Names and addresses here are
    # synthetic benchmark personas; preserve them because they are task inputs.
    files=sorted((ROOT/'p5/results_full').glob('*.json'))
    assert len(files)==2
    for file in files:
        rows=json.loads(file.read_text())
        calls=sum(bool(m.get('tool_calls')) for r in rows for m in r['traj'] if m.get('role')=='assistant')
        arm='documented' if calls==0 else 'repaired'
        assert len(rows)==115
        exported=[{k:r[k] for k in ['task_id','reward','info','traj','trial']} for r in sorted(rows,key=lambda r:r['task_id'])]
        write(f'tau/{arm}.jsonl',exported)
        summary[f'tau_{arm}']={'tasks':len(rows),'parsed_assistant_turns':calls}
    # All first HTTP responses in each factorial arm, without host/request headers.
    for arm in ['hermes','hermes_dedtpl','dedparser_doctpl','repaired']:
        file=ROOT/f'runs/p1_2x2/logs/bfcl_{arm}.jsonl'
        rows=[json.loads(l) for l in file.read_text().splitlines()]
        rows=[r for r in rows if r['request_index']==0]
        assert len(rows)==200 and len({r['case_id'] for r in rows})==200
        keep=['case_id','request_index','status','tool_choice','n_tools_offered',
              'tool_names_offered','parsed_tool_calls','content','finish_reason']
        write(f'bfcl_factorial/{arm}.jsonl',[{k:r[k] for k in keep} for r in sorted(rows,key=lambda r:r['case_id'])])
        summary[f'bfcl_{arm}']={'first_requests':len(rows),'parsed':sum(bool(r['parsed_tool_calls']) for r in rows)}
    # Deterministic unconditioned sample: ten lowest (sample_index, rollout_n)
    # trajectory keys from training step 1 in each arm, all relevant events.
    source=ROOT/'p3/results/formal_20260916_a/raw/runs/p3/p3-formal-20260916-a'
    fields=['kind','step','sample_index','rollout_n','validate','parser','accepted',
            'accepted_names','tight','text','text_chars','text_sha256','tool_name',
            'assistant_turn','has_code','arguments_sha256','observation',
            'observation_chars','observation_sha256']
    for arm in ['broken','repaired']:
        grouped=collections.defaultdict(list)
        for file in sorted((source/arm/'events').glob('*.jsonl')):
            for line in file.open():
                r=json.loads(line)
                if r.get('step')!=1 or r.get('validate') is True: continue
                if r.get('kind') not in ['extract','call_tool','execute','call_tool_result']:continue
                key=(r['sample_index'],r['rollout_n'])
                clean={k:r[k] for k in fields if k in r}
                for textkey in ['text','observation']:
                    if textkey in clean:
                        assert hashlib.sha256(clean[textkey].encode()).hexdigest()==clean[textkey+'_sha256']
                if 'request_id' in r:
                    clean['request_id']=hashlib.sha256(str(r['request_id']).encode()).hexdigest()[:24]
                grouped[key].append(clean)
        selected=sorted(grouped)[:10]
        assert len(selected)==10
        exported=[dict(arm=arm,trajectory=f'{arm}-{i:02}',event_index=j,**r)
                  for i,key in enumerate(selected) for j,r in enumerate(grouped[key])]
        write(f'rl/{arm}.jsonl',exported)
        summary[f'rl_{arm}']={'trajectories':10,'events':len(exported),
            'selection':'ten lowest (sample_index, rollout_n) at training step 1; no outcome filtering'}
    (OUT/'inventory.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
