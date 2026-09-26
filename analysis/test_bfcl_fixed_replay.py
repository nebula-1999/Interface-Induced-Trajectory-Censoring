import unittest
from bfcl_fixed_replay import recover, structural_schema

class ReplayTests(unittest.TestCase):
    def test_no_repair(self):
        self.assertEqual(recover('{"name":"x","arguments":{"a":NaN}}'),[])
        self.assertEqual(recover('{"name":"x","arguments":oops}'),[])
    def test_all_calls_not_gold_selection(self):
        x='[{"name":"wrong","arguments":{}},{"name":"right","arguments":{}}]'
        self.assertEqual([c['name'] for c in recover(x)],['wrong','right'])
    def test_nested_and_string_arguments(self):
        self.assertEqual(recover('{"function":{"name":"x","arguments":"{}"}}'),
                         [{'name':'x','arguments':{}}])
    def test_schema(self):
        s={'type':'object','required':['n'],'properties':{'n':{'type':'integer'}}}
        self.assertTrue(structural_schema({'n':1},s))
        for x in [{},{'n':True},{'n':'1'},{'n':1,'extra':2}]:self.assertFalse(structural_schema(x,s))

if __name__=='__main__':unittest.main()
