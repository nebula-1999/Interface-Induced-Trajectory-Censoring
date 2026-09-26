#!/usr/bin/env python3
"""Audit frozen native-FC outcomes and compute post-hoc paired contrasts on CPU."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from submission_statistics import paired_interval

ARMS = ['base', 'broken', 'repaired']
DEFECT_IDS = {'HumanEval/32', 'Mbpp/599'}
EXPECTED = {'base': None,
 'broken': '84251925cbf53418e9404869f58cf9f5e80a71d98aace7763273368da2f24217',
 'repaired': '07a7cd9627798d7e94bba9afe8630b5c0c316b6353a5c34bbb151e9de088c779'}


def reduce_archive(root):
    rows=[]; configs=[]
    for arm in ARMS:
        p=root/arm; c=json.loads((p/'configuration.json').read_text()); configs.append(c)
        raw=[json.loads(s) for s in (p/'trajectories.jsonl').read_text().splitlines()]
        assert not (p/'INVALID.json').exists()
        assert len(raw)==542==json.loads((p/'COMPLETE.json').read_text())['n']
        assert c['identity']['runtime_tensor_sha256']==EXPECTED[arm]
        assert c['arm']==arm and c['smoke'] is False
        assert c['ids']==[x['task_id'] for x in raw] and len(set(c['ids']))==542
        for r in raw:
            for e in r['events']:
                assert hashlib.sha256(e['raw'].encode()).hexdigest()==e['raw_sha256']
                if e['executed']:
                    assert any(m['role']=='tool' and m.get('tool_call_id')==e['request_id']
                               and m['content']==e['observation'] for m in r['messages'])
            assert r['tool_executions']==sum(e['executed'] for e in r['events'])
            term=r['terminal_result']
            assert r['final_pass']==bool(term and term['all_passed'])
            rows.append(dict(arm=arm,task_id=r['task_id'],source=r['source'],
                final_pass=r['final_pass'],initiated=r['tool_executions']>0,
                rescued=r['rescued'],
                final_after_failed_tool=r['first_tool_test_pass'] is False and r['final_pass'],
                executions=r['tool_executions'],
                generations=len(r['events']),
                accepted_calls=sum(e['accepted_calls'] for e in r['events']),
                ignored_calls=sum(e['ignored_calls'] for e in r['events']),
                length_stops=sum(e['finish_reason']=='length' for e in r['events']),
                terminal_status=term['status'] if term else 'no_submission',
                execution_statuses=[e['result']['status'] for e in r['events'] if e['executed']]))
    assert all(c['ids']==configs[0]['ids'] and c['sources']==configs[0]['sources'] for c in configs)
    return rows


def analyze(rows):
    arms={a:{r['task_id']:r for r in rows if r['arm']==a} for a in ARMS}
    assert len(rows)==1626
    assert all(len(v)==542 and v.keys()==arms['base'].keys() for v in arms.values())
    summaries={}
    for a,rs in arms.items():
        vals=list(rs.values())
        summaries[a]={k:sum(r[k] for r in vals) for k in
            ['final_pass','initiated','rescued','executions','generations','accepted_calls','ignored_calls','length_stops']}
        summaries[a].update(n=len(vals),terminal_status=dict(Counter(r['terminal_status'] for r in vals)),
            execution_status=dict(Counter(s for r in vals for s in r['execution_statuses'])),
            timeout_ids=[r['task_id'] for r in vals if r['terminal_status']=='timeout'])
    sensitivity={a:dict(n=540,final_pass=sum(r['final_pass'] for k,r in rs.items() if k not in DEFECT_IDS),
        final_after_failed_tool=sum(r['final_after_failed_tool'] for r in rs.values()),
        known_defect_outcomes={k:rs[k]['final_pass'] for k in sorted(DEFECT_IDS)}) for a,rs in arms.items()}
    contrasts=[]
    for a,b in [('base','broken'),('base','repaired'),('broken','repaired')]:
        for metric in ['final_pass','initiated','rescued']:
            gains=[k for k in arms[a] if not arms[a][k][metric] and arms[b][k][metric]]
            losses=[k for k in arms[a] if arms[a][k][metric] and not arms[b][k][metric]]
            c=paired_interval(len(gains),len(losses),542)
            c.update(left=a,right=b,metric=metric,gain_ids=gains,loss_ids=losses,
                     bonferroni9_p=min(1.,9*c['mcnemar_p']))
            contrasts.append(c)
    return dict(status='PASS',scope='Post-hoc native-FC evaluation; no new training seed',
        family='Nine exploratory contrasts (three paired weights x three binary endpoints); separate from historical 27-test audit',
        interval_scope='Pointwise conservative item-pair intervals; exclude training-seed and inference-retest variability',
        timeout_policy='Retain all 542 tasks; terminal timeouts are unsuccessful outcomes',
        arms=summaries,contrasts=contrasts,
        descriptive_sensitivity=sensitivity,
        sensitivity_note='540-item final success excludes two historically flagged defects; final_after_failed_tool remains on all 542 items. Descriptive only, no added hypothesis tests.')


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--archive',type=Path);g.add_argument('--outcomes',type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    rows=reduce_archive(a.archive) if a.archive else json.loads(a.outcomes.read_text())
    report=analyze(rows);a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'outcomes.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.out/'statistics.json').write_text(json.dumps(report,indent=2)+'\n')
    if a.archive:
        configs={}
        for arm in ARMS:
            c=json.loads((a.archive/arm/'configuration.json').read_text())
            c['identity']={'weight':arm,'runtime_tensor_sha256':c['identity']['runtime_tensor_sha256']}
            configs[arm]=c
        (a.out/'configurations.json').write_text(json.dumps(configs,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report['arms'],indent=2))


if __name__=='__main__':main()
