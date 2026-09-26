#!/usr/bin/env python3
"""Export outcome-only and event-only evidence; never export model/task text."""
import gzip
import hashlib
import json
from pathlib import Path
from p3_final_evidence import STEPS, final_eval_path, row_outcome, audit_eval, audit_events

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'p3/results/formal_20260916_a/raw/runs/p3/p3-formal-20260916-a'
OUT = ROOT / 'submission_packages/p3_compact'
RANGES = {'broken': [1, 150], 'repaired': [1, 90], 'repaired_resume_00090': [91, 150]}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation, _ = audit_eval(SOURCE)
    events, _ = audit_events(SOURCE)  # validates retained raw text hashes before reduction
    records = []
    for arm in ('broken', 'repaired'):
        for step in STEPS:
            path = final_eval_path(SOURCE, arm, step)
            for line in path.read_text().splitlines():
                r = json.loads(line)
                first, final, rescue = row_outcome(r)
                task = hashlib.sha256(json.dumps([r['channel'], r['task_id']]).encode()).hexdigest()[:24]
                records.append(dict(arm=arm, step=step, channel=r['channel'], task_id=task,
                                    first_ok=first, final_ok=final, rescued=rescue,
                                    turns=len(r['turns'])))
    (OUT/'evaluation_outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    fields = ('kind', 'step', 'sample_index', 'rollout_n', 'validate', 'assistant_turn',
              'tight', 'accepted', 'text_chars', 'observation_chars')
    with gzip.GzipFile(filename=str(OUT/'events_compact.jsonl.gz'), mode='wb', mtime=0) as out:
        for part, (lo, hi) in RANGES.items():
            for file in sorted((SOURCE/part/'events').glob('events.*.jsonl')):
                for index, line in enumerate(file.read_text().splitlines()):
                    r = json.loads(line)
                    if r.get('kind') not in ('extract', 'call_tool', 'execute', 'call_tool_result'):
                        continue
                    compact = {k:r[k] for k in fields if k in r}
                    step = r.get('step')
                    included = step is not None and lo <= step <= hi and r.get('validate') is not True
                    compact.update(source_part=part, source_file=file.name, source_line=index+1,
                                   final_lineage=included,
                                   exclusion_reason=None if included else 'outside_accepted_training_segment')
                    if 'request_id' in r:
                        compact['request_id'] = hashlib.sha256(str(r['request_id']).encode()).hexdigest()[:24]
                    out.write((json.dumps(compact)+'\n').encode())
    (OUT/'lineage.json').write_text(json.dumps(dict(accepted_step_ranges=RANGES,
        note='Original repaired steps 91-116 are excluded. Resumed evaluation at step 90 is not selected.'), indent=2)+'\n')
    expected = dict(curves={a:{s:{c:{k:v for k,v in d.items() if k in ('n','turn1_pass','final_pass','rescued')}
                           for c,d in cs.items()} for s,cs in ss.items()} for a,ss in evaluation['curves'].items()},
                    paired_step150=evaluation['paired_step150'],
                    events={a:{k:v for k,v in d.items() if k in ('generation_events','tight_events','accepted_calls',
                         'accepted_generation_events','dispatch_events','execute_events','observation_events')}
                            for a,d in events.items()})
    (OUT/'expected.json').write_text(json.dumps(expected,indent=2)+'\n')
    print(f'Exported {len(records)} outcomes and compact event archive to {OUT}')

if __name__ == '__main__': main()
