#!/usr/bin/env python3
"""No model load: real tokenizer/schema and trusted toy executor acceptance."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from native_fc_eval import TOOLS,evaluate_task

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
    a=parser.parse_args()
    from transformers import AutoTokenizer
    from sandbox import run_many
    tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    outputs=iter([json.dumps({'name':'run_tests','arguments':{'code':'def f():\n    return 1'}}),'Done.'])
    renders=[]
    def generate(messages):
        rendered=tokenizer.apply_chat_template(messages,tools=TOOLS,tokenize=False,add_generation_prompt=True)
        assert 'run_tests' in rendered
        renders.append(rendered)
        return next(outputs),'stop'
    def test(code):
        result=run_many([(code,'assert f() == 1\n')],workers=1,mode='script',timeout=30)[0]
        return dict(passed=result.passed,total=result.total,all_passed=result.all_passed,
                    status=result.status,stderr=result.stderr)
    result=evaluate_task('fixture','Implement f returning 1','Python assistant',generate,test)
    assert result['tool_executions']==1 and result['final_pass']
    assert '<tool_response>' in renders[1]
    assert result['messages'][-2]['role']=='tool'
    print(json.dumps(dict(status='CPU_SMOKE_PASS',tool_executions=1,terminal_success=True,
          tool_response_in_render=True,model_loaded=False)))

if __name__=='__main__':main()
