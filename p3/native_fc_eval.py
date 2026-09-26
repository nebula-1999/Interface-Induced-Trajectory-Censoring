#!/usr/bin/env python3
"""Common raw-generation FC evaluator. CLI requires explicit GPU opt-in.

Mockable controller: only accepted calls execute during interaction; terminal
scoring never feeds results back. This is post-hoc held-out FC, not training.
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qwen_tools_parser import extract_calls
from code_tool_core import extract_code, format_observation

TOOLS=[{'type':'function','function':{'name':'run_tests',
 'description':'把你写的 Python 代码交给测试运行，返回通过情况与报错。',
 'parameters':{'type':'object','properties':{'code':{'type':'string',
 'description':'完整的 Python 代码（函数定义及其依赖的 import）'}},'required':['code']}}}]

def accepted_calls(raw):
    _,parsed=extract_calls(raw)
    accepted=[]
    for name,args in parsed:
        if name!='run_tests':continue
        try:obj=json.loads(args)
        except (ValueError,TypeError):continue
        code=obj.get('code') if isinstance(obj,dict) else None
        if isinstance(code,str) and code.strip():accepted.append((name,code))
    return parsed,accepted

def final_code(raw):
    code=extract_code(raw)
    if '```' in raw:return code
    try:tree=ast.parse(code)
    except SyntaxError:return None
    return code if any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for n in tree.body) else None

def evaluate_task(task_id,question,system,generate,test,max_turns=4):
    messages=[{'role':'system','content':system},{'role':'user','content':question}]
    events=[];last_code=None;submission_source=None;first_test=None;rescued=False
    for turn in range(1,max_turns+1):
        raw,finish=generate(messages)
        parsed,accepted=accepted_calls(raw)
        event=dict(turn=turn,raw=raw,raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            finish_reason=finish,parsed_candidates=len(parsed),accepted_calls=len(accepted),
            ignored_calls=max(0,len(accepted)-1),executed=False)
        messages.append({'role':'assistant','content':raw})
        if not accepted:
            candidate=final_code(raw)
            if candidate is not None:last_code=candidate;submission_source='final_text'
            events.append(event);break
        last_code=extract_code(accepted[0][1]);submission_source='tool_call'
        result=test(last_code)
        if first_test is None:first_test=bool(result['all_passed'])
        elif not first_test and result['all_passed']:rescued=True
        observation=format_observation(result['passed'],result['total'],result['status'],result['stderr'],1200)
        call_id=f'{task_id}:turn:{turn}'
        event.update(executed=True,code=last_code,request_id=call_id,result=result,observation=observation)
        messages.append({'role':'tool','name':'run_tests','tool_call_id':call_id,'content':observation})
        events.append(event)
    # Independent terminal test: no new user/tool message and no further generation.
    terminal=test(last_code) if last_code is not None else None
    return dict(task_id=task_id,events=events,messages=messages,final_code=last_code,
        submission_source=submission_source,terminal_result=terminal,
        final_pass=bool(terminal and terminal['all_passed']),rescued=rescued,
        first_tool_test_pass=first_test,tool_executions=sum(e['executed'] for e in events))

class HFPolicy:
    def __init__(self,base,adapter,max_tokens=1024):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU generation fallback')
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        self.torch=torch;self.max_tokens=max_tokens
        for name,expected in [('config.json','c0242402ad6a13b331ea320feea8c7e3776ffb7a4eff0757b9cd667e116d9a28'),
                              ('tokenizer_config.json','959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4')]:
            assert hashlib.sha256((Path(base)/name).read_bytes()).hexdigest()==expected
        self.tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,
            torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
        self.identity={'base':str(base),'adapter':None,'runtime_tensor_sha256':None}
        if adapter:
            from peft import PeftModel
            record=json.loads((adapter/'VERIFIED.json').read_text())
            for name,expected in record['files_sha256'].items():
                assert hashlib.sha256((adapter/name).read_bytes()).hexdigest()==expected
            assert hashlib.sha256((Path(base)/'config.json').read_bytes()).hexdigest()==record['base_config_sha256']
            self.model=PeftModel.from_pretrained(self.model,adapter,is_trainable=False,autocast_adapter_dtype=False)
            named=dict(self.model.named_parameters());keys=sorted(k for k in named if '.lora_' in k)
            h=hashlib.sha256()
            assert len(keys)==392
            for k in keys:h.update(named[k].detach().float().cpu().contiguous().numpy().tobytes())
            assert h.hexdigest()==record['tensor_sha256'],'Loaded adapter differs from verified checkpoint'
            for module in self.model.modules():
                if hasattr(module,'scaling') and isinstance(module.scaling,dict):
                    assert module.scaling['default']==1.0
            self.identity.update(adapter=str(adapter),runtime_tensor_sha256=h.hexdigest())
        self.model.eval()
    def __call__(self,messages):
        torch=self.torch
        encoded=self.tokenizer.apply_chat_template(messages,tools=TOOLS,tokenize=True,
            add_generation_prompt=True,return_tensors='pt',return_dict=True)
        encoded={k:v.to(self.model.device) for k,v in encoded.items()}
        length=encoded['input_ids'].shape[-1]
        if length+self.max_tokens>8192:raise RuntimeError('Context budget exceeded; do not truncate silently')
        with torch.inference_mode():
            output=self.model.generate(**encoded,do_sample=False,max_new_tokens=self.max_tokens,
                                       pad_token_id=self.tokenizer.eos_token_id)
        new=output[0,length:]
        return self.tokenizer.decode(new,skip_special_tokens=True),('length' if len(new)>=self.max_tokens else 'stop')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',required=True,type=Path)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--arm',choices=['base','broken','repaired'],required=True)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--allow-gpu',action='store_true')
    a=p.parse_args()
    if not a.allow_gpu:p.error('Explicit --allow-gpu required; no inference started')
    if (a.arm=='base') != (a.adapter is None):p.error('Only base arm may omit adapter')
    if a.limit<0:p.error('limit must be nonnegative')
    if a.limit>542:p.error('limit exceeds frozen task count')
    expected={'broken':'84251925cbf53418e9404869f58cf9f5e80a71d98aace7763273368da2f24217',
              'repaired':'07a7cd9627798d7e94bba9afe8630b5c0c316b6353a5c34bbb151e9de088c779'}
    if a.adapter:
        record=json.loads((a.adapter/'VERIFIED.json').read_text())
        if record['tensor_sha256']!=expected[a.arm]:p.error('Adapter does not match arm identity')
    if a.out.exists():raise FileExistsError('Use a new run directory; never mix or overwrite runs')
    assert hashlib.sha256((ROOT/'qwen_tools_parser.py').read_bytes()).hexdigest()==\
        '494b2c77d72a7351f6becfb1ae45e0e92e619b1f5f47de7f70a725b1837199c2'
    from eval_decompose import question_of
    from sandbox import build_evalplus,run_tests
    manifest=json.loads(a.manifest.read_text())
    canonical=json.dumps(manifest,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    digest=hashlib.sha256(canonical.encode()).hexdigest()
    assert digest=='17d4fc01000e556ad82546942c25cffbbbda85bebc5ab853e59edab2c2cbcb20'
    records=manifest['records'];assert len(records)==542
    if a.limit:records=records[:a.limit//2]+records[-(a.limit-a.limit//2):]
    policy=HFPolicy(a.base,a.adapter)
    a.out.mkdir(parents=True,exist_ok=False)
    config=dict(arm=a.arm,manifest_sha256=digest,identity=policy.identity,
        protocol='raw-generation/native-FC/controller-v1',tools=TOOLS,max_turns=4,max_tokens=1024,
        temperature=0,max_context=8192,dispatch_cap=1,observation_chars=1200,
        ids=[r[1] for r in records],smoke=bool(a.limit),
        sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                 ['p3/native_fc_eval.py','qwen_tools_parser.py','code_tool_core.py','sandbox.py']})
    import importlib.metadata
    config['packages']={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','safetensors']}
    (a.out/'configuration.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
    start=time.time()
    try:
        with (a.out/'trajectories.jsonl').open('x') as out:
            for source,tid,rec in records:
                def test(code):
                    solution,tests=build_evalplus(rec,solution=code)
                    r=run_tests(solution,tests,mode='script',timeout=30.0)
                    return dict(passed=r.passed,total=r.total,all_passed=r.all_passed,status=r.status,stderr=r.stderr)
                row=evaluate_task(tid,question_of(rec),manifest['system'],policy,test)
                row.update(source=source,arm=a.arm)
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
        (a.out/'COMPLETE.json').write_text(json.dumps(dict(n=len(records),elapsed_s=time.time()-start)))
    except Exception as error:
        (a.out/'INVALID.json').write_text(json.dumps(dict(error=repr(error))))
        raise

if __name__=='__main__':main()
