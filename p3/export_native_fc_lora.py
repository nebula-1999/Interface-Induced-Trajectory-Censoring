#!/usr/bin/env python3
"""CPU-only export from a trusted, single-rank verl checkpoint. Never overwrite.

Writes an incomplete marker first; only round-trip verified exports get VERIFIED.json.
Run with OMP_NUM_THREADS=1 on no-GPU instances. Original weights are read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

KEY = re.compile(r'^(base_model\.model\.model\.layers\.\d+\.(?:self_attn|mlp)\.[a-z_]+\.lora_[AB])\.default\.weight$')
MODULES = ['down_proj','gate_proj','k_proj','o_proj','q_proj','up_proj','v_proj']

def export_key(key):
    m=KEY.fullmatch(key)
    if not m:raise ValueError(f'Unexpected adapter key: {key}')
    return m[1]+'.weight'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Refusing to overwrite any existing export')
    import torch
    from safetensors.torch import save_file,load_file
    from peft import LoraConfig
    torch.set_num_threads(1)
    meta=json.loads((a.actor/'lora_train_meta.json').read_text())
    assert meta['r']==32 and meta['lora_alpha']==32 and meta['task_type']=='CAUSAL_LM'
    source=a.actor/'model_world_size_1_rank_0.pt'
    state=torch.load(source,map_location='cpu',mmap=True,weights_only=True)
    keys=sorted(k for k in state if '.lora_' in k)
    assert len(keys)==392
    tensors={};h=hashlib.sha256();count=0
    for key in keys:
        t=state[key].detach()
        assert t.ndim==2 and (t.shape[0] if '.lora_A.' in key else t.shape[1])==32
        assert t.dtype==torch.bfloat16 and bool(torch.isfinite(t).all())
        h.update(t.float().contiguous().numpy().tobytes());count+=t.numel()
        tensors[export_key(key)]=t.contiguous().clone()
    assert h.hexdigest()==a.expected_sha256 and count==80740352
    assert sorted({k.split('.')[-3] for k in tensors})==MODULES
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'INCOMPLETE').write_text('Do not load before VERIFIED.json exists.\n')
    config=LoraConfig(r=32,lora_alpha=32,target_modules=MODULES,bias='none',
        task_type='CAUSAL_LM',inference_mode=True,lora_dropout=0.0,use_rslora=False,
        use_dora=False,base_model_name_or_path=str(a.base))
    config.save_pretrained(a.out)
    save_file(tensors,str(a.out/'adapter_model.safetensors'))
    restored=load_file(str(a.out/'adapter_model.safetensors'),device='cpu')
    assert set(restored)==set(tensors)
    digest=hashlib.sha256()
    for key in keys:
        t=restored[export_key(key)]
        assert torch.equal(t,tensors[export_key(key)])
        digest.update(t.float().contiguous().numpy().tobytes())
    assert digest.hexdigest()==a.expected_sha256
    cfg=LoraConfig.from_pretrained(a.out)
    assert cfg.r==32 and cfg.lora_alpha==32 and not cfg.use_rslora and not cfg.use_dora
    report=dict(status='CPU_ROUNDTRIP_PASS',source=str(source),tensor_sha256=digest.hexdigest(),
        tensors=392,parameters=count,base=str(a.base),
        files_sha256={n:hashlib.sha256((a.out/n).read_bytes()).hexdigest() for n in
                      ['adapter_config.json','adapter_model.safetensors']},
        base_config_sha256=hashlib.sha256((a.base/'config.json').read_bytes()).hexdigest(),
        runtime_loading_verified=False)
    (a.out/'VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'INCOMPLETE').unlink()
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
