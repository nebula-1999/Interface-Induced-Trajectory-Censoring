#!/usr/bin/env python3
"""Content and aggregate checks for the supplied raw subset; no code execution."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'mechanism_evidence'
if not BUNDLE.exists(): BUNDLE=ROOT/'submission_packages/mechanism_evidence'

def rows(path):return [json.loads(l) for l in path.read_text().splitlines()]

def main():
    for arm,n in [('hermes',0),('hermes_dedtpl',0),('dedparser_doctpl',0),('repaired',196)]:
        records=rows(BUNDLE/f'bfcl_factorial/{arm}.jsonl')
        assert len(records)==len({r['case_id'] for r in records})==200
        assert sum(bool(r['parsed_tool_calls']) for r in records)==n
    for arm,parsed,solved in [('documented',0,7),('repaired',636,10)]:
        records=rows(BUNDLE/f'tau/{arm}.jsonl')
        assert len(records)==len({r['task_id'] for r in records})==115
        assert sum(r['reward']==1 for r in records)==solved
        assert sum(bool(m.get('tool_calls')) for r in records for m in r['traj'] if m['role']=='assistant')==parsed
    # Exactly the classifier used by the retained rollout instrumentation.
    tight=re.compile(r'"name"\s*:\s*"run_tests".{0,200}?"arguments"\s*:\s*\{.{0,80}?"code"\s*:\s*"(.{0,4000}?)"\s*\}',re.S)
    real=re.compile(r'\\n|def |class |return |import |lambda ')
    for arm in ['broken','repaired']:
        records=rows(BUNDLE/f'rl/{arm}.jsonl')
        assert len({r['trajectory'] for r in records})==10
        for r in records:
            for k in ['text','observation']:
                if k in r:assert hashlib.sha256(r[k].encode()).hexdigest()==r[k+'_sha256']
            if r['kind']=='extract':
                m=tight.search(r['text'])
                assert bool(m and real.search(m.group(1)))==r['tight']
    print('Raw subset verified: 800 BFCL responses, 230 tau trajectories, 20 RL trajectories.')

if __name__=='__main__':main()
