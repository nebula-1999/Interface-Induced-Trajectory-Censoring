#!/usr/bin/env python3
"""Verify reduced records without private archives or third-party dependencies.

This verifies recorded labels, not their validity against omitted raw emissions.
Extraction can be joined to dispatch only at trajectory granularity.
"""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

def verify(root):
    expected = json.loads((root/'expected.json').read_text())
    lineage = json.loads((root/'lineage.json').read_text())['accepted_step_ranges']
    groups = defaultdict(dict)
    for line in (root/'evaluation_outcomes.jsonl').read_text().splitlines():
        r = json.loads(line); key = (r['channel'], r['task_id'])
        group = groups[r['arm'], str(r['step'])]
        assert key not in group, 'duplicate evaluation key'
        assert r['rescued'] == (r['final_ok'] and not r['first_ok'])
        group[key] = r
    assert set(groups) == {(a,str(s)) for a in ('broken','repaired') for s in (0,30,60,90,120,150)}
    baseline_keys = set(groups['broken','0'])
    for (arm, step), rows in groups.items():
        assert set(rows) == baseline_keys
        for channel, want in expected['curves'][arm][step].items():
            selected = [r for (c,_),r in rows.items() if c == channel]
            actual = dict(n=len(selected),turn1_pass=sum(r['first_ok'] for r in selected),
                          final_pass=sum(r['final_ok'] for r in selected),rescued=sum(r['rescued'] for r in selected))
            assert actual == want, (arm,step,channel,actual,want)
    for channel,want in expected['paired_step150'].items():
        keys = [k for k in baseline_keys if channel == 'all' or k[0] == channel]
        gain=loss=0
        for k in keys:
            a,b = [groups[arm,'150'][k]['final_ok'] for arm in ('broken','repaired')]
            gain += b and not a; loss += a and not b
        assert dict(n=len(keys),repaired_gains_vs_broken=gain,repaired_losses_vs_broken=loss,
                    discordant_total=gain+loss) == want
    totals = defaultdict(Counter); joins = defaultdict(lambda:defaultdict(Counter))
    accepted = defaultdict(Counter); dispatched = defaultdict(Counter); excluded = Counter()
    source_keys = set()
    with gzip.open(root/'events_compact.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line); part=r['source_part']; step=r.get('step'); lo,hi=lineage[part]
            source_key=(part,r['source_file'],r['source_line'])
            assert source_key not in source_keys; source_keys.add(source_key)
            keep=step is not None and lo<=step<=hi and r.get('validate') is not True
            assert keep == r['final_lineage']
            if not keep:
                excluded[part]+=1; continue
            arm='broken' if part=='broken' else 'repaired'
            trajectory=(part,step,r['sample_index'],r['rollout_n'])
            kind=r['kind']; t=totals[arm]
            if kind=='extract':
                n=int(r.get('accepted',0)); t['generation_events']+=1
                t['tight_events']+=bool(r.get('tight')); t['accepted_calls']+=n
                t['accepted_generation_events']+=n>0
                if n: accepted[arm][trajectory]+=1
            else:
                t[{'call_tool':'dispatch_events','execute':'execute_events','call_tool_result':'observation_events'}[kind]]+=1
                joins[arm][kind][(*trajectory,r['request_id'],r['assistant_turn'])]+=1
                if kind=='call_tool': dispatched[arm][trajectory]+=1
    for arm,want in expected['events'].items():
        assert dict(totals[arm]) == want, (arm,totals[arm],want)
        j=joins[arm]; assert j['call_tool']==j['execute']==j['call_tool_result']
        assert all(n==1 for n in j['call_tool'].values())
        assert accepted[arm]==dispatched[arm]
    print(json.dumps(dict(status='PASS',evaluation_rows=sum(map(len,groups.values())),
                          events=dict(totals),excluded_records=dict(excluded)),indent=2))

if __name__=='__main__':
    verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'p3/compact')
