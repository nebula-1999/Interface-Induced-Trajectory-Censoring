import unittest
from native_fc_preflight import audit, canonical_sha

class OfflineAuditTests(unittest.TestCase):
    def test_canonical_manifest_hash(self):
        self.assertEqual(canonical_sha({'a':1,'b':2}),canonical_sha({'b':2,'a':1}))
    def test_not_launch_authorization(self):
        report=audit()
        self.assertEqual(report['status'],'NOT_READY_FOR_GPU')
        self.assertEqual(report['held_out_tasks'],542)
        self.assertEqual(report['repair_tasks'],454)
        self.assertFalse(any(r['verified_now'] for r in report['endpoints'].values()))
        self.assertIn('repaired_resume_00090',report['endpoints']['repaired']['path'])

if __name__=='__main__':unittest.main()
