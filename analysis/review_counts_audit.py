#!/usr/bin/env python3
"""Read-only recheck of quantities flagged by the September LLM feedback."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    result = {'turn_accounting': {}, 'prompt_provenance': {}, 'llama_parsing': {}, 'cap_hits': {}}
    for name in ('v3_Qwen7B_react','v14_Qwen7B_react','v5_Qwen7B_fc_intent','v6b_Qwen7B_fc_plugin'):
        rows=load(ROOT/'runs/final'/f'traj_{name}.jsonl')
        result['turn_accounting'][name]=dict(n=len(rows),
            mean_assistant_records=sum(len(r['turns']) for r in rows)/len(rows),
            mean_code_tested_steps=sum(r['n_turns'] for r in rows)/len(rows),
            code_tested_steps_ge2=sum(r['n_turns']>=2 for r in rows),
            assistant_records_ge2=sum(len(r['turns'])>=2 for r in rows),
            zero_code_tested_steps=sum(r['n_turns']==0 for r in rows))
    for name in ('a14_Qwen1.5B_fc_mandatory','roleprobe_Qwen1.5B_fc_roledisambig'):
        rows=load(ROOT/'runs/final'/f'traj_{name}.jsonl')
        result['prompt_provenance'][name]=dict(recorded_conditions=[list(k) for k in
            sorted(set((r['strength'],r['sys_file']) for r in rows))],
            override_file_present=(ROOT/'runs/final'/rows[0]['sys_file']).is_file())
    for name in ('v3_Llama8B_fc','v6_Llama8B_fc_strict'):
        rows=load(ROOT/'runs/final'/f'traj_{name}.jsonl')
        result['llama_parsing'][name]=dict(n=len(rows),parsed=sum(
            r['turns'][0].get('parse_mode')=='fc_tool_call' for r in rows))
    for scale in ('1.5B','3B','7B','14B','32B'):
        rows=load(ROOT/'runs/final'/f'traj_v5_Qwen{scale}_fc_intent.jsonl')
        result['cap_hits'][scale]=dict(n=len(rows),at_storage_cap=sum(
            len(r['turns'][0].get('raw_output') or '')==4000 for r in rows))
    rows=load(ROOT/'p3/results/formal_20260916_a/raw/runs/p3/p3-formal-20260916-a/broken/eval/step_00000.jsonl')
    result['eval_channel_sources']={f'{c}/{s}':n for (c,s),n in Counter(
        (r['channel'],r['source']) for r in rows).items()}
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
