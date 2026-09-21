import math
import unittest
from submission_statistics import cp, paired_interval


class IntervalTests(unittest.TestCase):
    def test_boundary_binomials(self):
        self.assertAlmostEqual(cp(0,10,.0125)[1],1-.0125**.1,places=12)
        self.assertAlmostEqual(cp(10,10,.0125)[0],.0125**.1,places=12)

    def test_swap_symmetry(self):
        a=paired_interval(0,3,542)['ci95']
        b=paired_interval(3,0,542)['ci95']
        self.assertEqual(a,[-b[1],-b[0]])

    def test_reported_endpoint(self):
        r=paired_interval(0,3,542)
        self.assertEqual(r['mcnemar_p'],.25)
        self.assertEqual([round(100*x,2) for x in r['ci95']],[-1.79,.72])

    def test_small_multinomial_coverage(self):
        n=8
        bounds={(g,l):paired_interval(g,l,n)['ci95'] for g in range(n+1) for l in range(n+1-g)}
        for pg,pl in [(0,0),(.01,.1),(.1,.1),(.25,.5),(.5,.5),(1,0)]:
            coverage=0
            for (g,l),(lo,hi) in bounds.items():
                rest=n-g-l
                mass=math.comb(n,g)*math.comb(n-g,l)*pg**g*pl**l*(1-pg-pl)**rest
                if lo-1e-12<=pg-pl<=hi+1e-12:coverage+=mass
            self.assertGreaterEqual(coverage, .95-1e-12)


if __name__=='__main__':unittest.main()
