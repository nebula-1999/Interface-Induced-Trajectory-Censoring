#!/usr/bin/env python3
"""Offline readiness audit, never starts a server, inference, or training.

Optionally verify endpoint LoRA tensors on a CPU host with --verify-weights.
The stored SHA values are sorted float32 LoRA tensor digests, not file digests.
An audit PASS on saved metadata is not authorization to run a GPU experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'p3/results/formal_20260916_a'
RAW=ARCHIVE/'raw/runs/p3/p3-formal-20260916-a'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                    separators=(',',':')).encode()).hexdigest()

def tensor_digest(path):
    import torch
    # No-GPU instances can have 0.5 CPU despite exposing host-wide core counts.
    torch.set_num_threads(1)
    state=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    keys=sorted(k for k in state if '.lora_' in k)
    if not keys:raise ValueError('No LoRA tensors; possible base-only checkpoint')
    h=hashlib.sha256();count=0
    for k in keys:
        t=state[k].detach().float().contiguous()
        h.update(t.numpy().tobytes());count+=t.numel()
    return dict(sha256=h.hexdigest(),tensors=len(keys),parameters=count)

def audit(verify_weights=False,run_root=None):
    old=json.loads((ARCHIVE/'checkpoint_lora_audit.json').read_text())
    manifest=RAW/'eval_manifest.json'
    m=json.loads(manifest.read_text())
    records=m['records'];ids=[r[1] for r in records]
    assert len(ids)==len(set(ids))==542
    repair_ids=set(m['probes'])&set(ids)
    assert len(repair_ids)==454
    endpoints={}
    for arm,pair in [('broken','broken_120_to_150'),('repaired','repaired_90_to_150')]:
        p=old['pairs'][pair]
        source=Path(p['end_path'])
        if run_root:
            part='broken' if arm=='broken' else 'repaired_resume_00090'
            source=run_root/part/'ckpt/global_step_150/actor/model_world_size_1_rank_0.pt'
        found=source.is_file()
        row=dict(path=str(source),exists_on_this_host=found,
                 expected_lora_sha256=p['sha256_end'],historical_audit_date=old['audit_date'],
                 verified_now=False)
        if verify_weights and found:
            d=tensor_digest(source)
            assert d['sha256']==p['sha256_end'],f'{arm}: wrong LoRA digest'
            assert d['tensors']==392 and d['parameters']==80740352
            row.update(verified_now=True,observed=d)
        endpoints[arm]=row
    timings={}
    for arm,part,step in [('base','broken',0),('broken','broken',150),('repaired','repaired_resume_00090',150)]:
        complete=json.loads((RAW/part/f'eval/step_{step:05d}.complete.json').read_text())
        assert complete['manifest_sha256']==canonical_sha(m)
        timings[arm]=complete['metrics']['code/elapsed_s']
    return dict(status='NOT_READY_FOR_GPU',
        evaluation_manifest_sha256=canonical_sha(m),
        evaluation_manifest_file_sha256=sha(manifest),held_out_tasks=542,repair_tasks=454,
        historical_protocol=m['protocol'],historical_both_channel_seconds=timings,
        endpoints=endpoints,
        source_sha256={p:sha(ROOT/p) for p in ['qwen_tools_parser.py','chat_policy.py',
                        'code_eval_hook.py','code_tool.py','sandbox.py']},
        blockers=['Endpoint files must be reverified on their storage host.',
                  'CPU exports now exist; verify export receipts and actual GPU-loaded adapter tensors.',
                  'Native FC controller passes CPU tests; real GPU smoke and throughput remain unverified.',
                  'Runtime attestation plus parser/executor smoke test must pass before a full run.'],
        safety='No SSH, server startup, model execution, checkpoint deletion or training performed.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--verify-weights',action='store_true')
    p.add_argument('--run-root',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args();result=audit(a.verify_weights,a.run_root)
    text=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')
    raise SystemExit(2) # metadata audit must not be mistaken for launch acceptance
