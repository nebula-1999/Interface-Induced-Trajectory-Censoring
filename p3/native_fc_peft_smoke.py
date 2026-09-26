#!/usr/bin/env python3
"""Tiny randomly initialized CPU model only; verify installed PEFT load semantics."""
import json
import tempfile
import torch
from peft import LoraConfig,get_peft_model,PeftModel
from transformers import Qwen2Config,Qwen2ForCausalLM

def main():
    torch.set_num_threads(1);torch.manual_seed(7)
    config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                      num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
    base=Qwen2ForCausalLM(config)
    initial={k:v.detach().clone() for k,v in base.state_dict().items()}
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=2,target_modules=['q_proj','v_proj'],
                                       task_type='CAUSAL_LM',bias='none'))
    with torch.no_grad():
        for k,v in model.named_parameters():
            if '.lora_B.' in k:v.fill_(0.1)
    model.eval();x=torch.tensor([[1,2,3]])
    with torch.no_grad():
        expected=model(x).logits
        with model.disable_adapter():unadapted=model(x).logits
    assert not torch.allclose(expected,unadapted)
    with tempfile.TemporaryDirectory(prefix='native-fc-peft-smoke-') as d:
        model.save_pretrained(d)
        other=Qwen2ForCausalLM(config);other.load_state_dict(initial)
        loaded=PeftModel.from_pretrained(other,d,is_trainable=False,autocast_adapter_dtype=False)
        loaded.eval()
        with torch.no_grad():actual=loaded(x).logits
        assert torch.allclose(expected,actual,rtol=1e-6,atol=1e-6)
        src={k:v for k,v in model.named_parameters() if '.lora_' in k}
        dst={k:v for k,v in loaded.named_parameters() if '.lora_' in k}
        assert src.keys()==dst.keys() and all(torch.equal(src[k],dst[k]) for k in src)
    print(json.dumps(dict(status='TINY_CPU_PEFT_PASS',logits_changed_with_adapter=True,
                          roundtrip_logits_equal=True,real_checkpoint_loaded=False)))

if __name__=='__main__':main()
