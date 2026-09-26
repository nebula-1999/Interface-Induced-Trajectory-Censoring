import json
import unittest
from native_fc_eval import evaluate_task, accepted_calls
from export_native_fc_lora import export_key

def call(code='def f():\n    return 1',name='run_tests'):
    return json.dumps({'name':name,'arguments':{'code':code}})

class ControllerTests(unittest.TestCase):
    def run_task(self,outputs,passes=None):
        seen=[];tested=[];outputs=iter(outputs);passes=iter(passes or [True]*10)
        def generate(messages):seen.append(json.loads(json.dumps(messages)));return next(outputs),'stop'
        def test(code):
            tested.append(code);ok=next(passes)
            return dict(passed=int(ok),total=1,all_passed=ok,status='ok',stderr='')
        return evaluate_task('case','question','system',generate,test),seen,tested
    def test_plain_code_no_interactive_feedback(self):
        r,seen,tested=self.run_task(['```python\ndef f(): return 1\n```'])
        self.assertEqual(r['tool_executions'],0)
        self.assertEqual(len(tested),1) # terminal scoring only
        self.assertTrue(r['final_pass'])
        self.assertFalse(any(m['role']=='tool' for m in r['messages']))
    def test_tool_role_and_rescue(self):
        r,seen,tested=self.run_task([call('def f(): return 0'),call(),'Done.'],[False,True,True])
        self.assertEqual(r['tool_executions'],2)
        self.assertEqual(seen[1][-1]['role'],'tool')
        self.assertTrue(r['rescued'])
        self.assertEqual(len(tested),3)
        self.assertEqual(r['submission_source'],'tool_call')
    def test_cap(self):
        r,_,tested=self.run_task([call()+'\n'+call('def f(): return 2'),'Done.'])
        self.assertEqual(r['events'][0]['accepted_calls'],2)
        self.assertEqual(r['events'][0]['ignored_calls'],1)
        self.assertNotIn('return 2',tested[0])
    def test_invalid_calls_no_execution(self):
        for output in [call(name='not_allowed'),'{"name":"run_tests","arguments":broken}']:
            r,_,tested=self.run_task([output])
            self.assertEqual(r['tool_executions'],0)
            self.assertEqual(tested,[])
    def test_max_turns(self):
        r,seen,tested=self.run_task([call()]*4)
        self.assertEqual(len(seen),4)
        self.assertEqual(len(tested),5)
    def test_export_key(self):
        src='base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight'
        self.assertEqual(export_key(src),src.replace('.default',''))
        with self.assertRaises(ValueError):export_key('model.weight')

if __name__=='__main__':unittest.main()
