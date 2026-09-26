import copy
import json
from pathlib import Path
import unittest
from native_fc_results import analyze


class NativeFCResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=json.loads((Path(__file__).resolve().parents[1]/
            'submission_packages/native_fc/outcomes.json').read_text())

    def test_recorded_counts_and_pairing(self):
        r=analyze(self.rows)
        self.assertEqual([r['arms'][a]['final_pass'] for a in ['base','broken','repaired']],[239,248,248])
        c=next(c for c in r['contrasts'] if c['left']=='broken' and c['right']=='repaired' and c['metric']=='final_pass')
        self.assertEqual((c['gains'],c['losses'],c['mcnemar_p']),(16,16,1.0))
        self.assertAlmostEqual(c['ci95'][0],-c['ci95'][1])
        self.assertEqual(len(r['contrasts']),9)

    def test_no_timeout_exclusion(self):
        r=analyze(self.rows)
        self.assertEqual([r['arms'][a]['terminal_status']['timeout'] for a in ['base','broken','repaired']],[4,5,6])
        self.assertTrue(all(x['n']==542 for x in r['arms'].values()))

    def test_missing_or_duplicate_items_rejected(self):
        with self.assertRaises(AssertionError):analyze(self.rows[:-1])
        rows=copy.deepcopy(self.rows);rows[1]['task_id']=rows[0]['task_id']
        with self.assertRaises(AssertionError):analyze(rows)

    def test_sensitivity_and_final_submission_rescue(self):
        s=analyze(self.rows)['descriptive_sensitivity']
        self.assertEqual([s[a]['final_pass'] for a in ['base','broken','repaired']],[239,247,247])
        self.assertEqual([s[a]['final_after_failed_tool'] for a in ['base','broken','repaired']],[2,2,1])
        self.assertEqual([s[a]['known_defect_outcomes']['Mbpp/599'] for a in ['base','broken','repaired']],[False,True,True])


if __name__=='__main__':unittest.main()
