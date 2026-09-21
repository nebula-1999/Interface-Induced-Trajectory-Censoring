#!/usr/bin/env python3
"""Retrospective, explicit multiplicity audit; conservative paired risk interval.

No training-seed uncertainty is estimated. The binomial construction assumes
independent evaluation-item pairs. Exact marginal coverage + a union bound
gives at least 95% coverage for p_gain - p_loss, without a Wald approximation.
"""
import argparse
import json
import math
from pathlib import Path
from llama_paired_audit import FILES, load, exact_mcnemar


def binom_cdf(k, n, p):
    if p == 0:
        return 1.0
    if p == 1:
        return float(k >= n)
    return min(1.0, math.fsum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
        - math.lgamma(n - i + 1) + i * math.log(p) + (n-i) * math.log1p(-p))
        for i in range(k + 1)))


def invert_cdf(k, n, target):
    lo, hi = 0.0, 1.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp(k, n, tail):
    return (0.0 if k == 0 else invert_cdf(k-1, n, 1-tail),
            1.0 if k == n else invert_cdf(k, n, tail))


def paired_interval(gains, losses, n, alpha=0.05):
    assert 0 <= gains + losses <= n
    lg, ug = cp(gains, n, alpha/4)
    ll, ul = cp(losses, n, alpha/4)
    return {'n': n, 'gains': gains, 'losses': losses,
            'difference': (gains-losses)/n, 'ci95': [lg-ul, ug-ll],
            'method': 'Bonferroni combination of two 97.5% Clopper-Pearson intervals',
            'mcnemar_p': exact_mcnemar(gains, losses)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    arms = {k: load(args.root/'runs/final'/v) for k,v in FILES.items()}
    rows = []
    for left, right in [('terse','rich'),('terse','thought'),('terse','official'),
                        ('terse','strict'),('official','strict')]:
        for metric in ['wrong_tool','turn1_pass','final_pass']:
            a,b=arms[left],arms[right]
            assert a.keys()==b.keys()
            gains=sum(not a[k][metric] and b[k][metric] for k in a)
            losses=sum(a[k][metric] and not b[k][metric] for k in a)
            rows.append(dict(comparison=f'Llama {left} to {right}: {metric}', n=100,
                gains=gains, losses=losses, provenance='item-level runs/final; llama_paired_audit.py'))
    # Persisted paired counts, not rounded p-values. Direction is later arm gain/loss.
    extra = [
        ('Llama strict to ReAct final',100,27,8,'G decomposition'),
        ('Llama terse to ReAct final',100,38,7,'G decomposition'),
        ('Qwen Hermes to adapter final',100,16,7,'repair loop'),
        ('Qwen adapter to ReAct final',100,17,5,'G main table'),
        ('Qwen Hermes to adapter final (random)',300,43,25,'runs/p0/RESULT.md'),
        ('Qwen adapter to ReAct final (random)',300,59,18,'runs/p0/RESULT.md'),
        ('Llama strict to ReAct final (random)',300,51,31,'runs/p0/RESULT.md'),
        ('Qwen both-parsed FC to ReAct final',83,11,4,'G conditioned table; descriptive subset'),
        ('Llama both-parsed FC to ReAct final',96,25,7,'G conditioned table; descriptive subset'),
        ('tau documented to repaired success',115,3,0,'G tau table'),
        ('P3 broken to repaired final at step150',542,0,3,'final-lineage audit.json'),
        ('P3 repaired step0 to step150 final',542,1,4,'final-lineage audit.json'),
    ]
    rows += [dict(comparison=s,n=n,gains=g,losses=l,provenance=src) for s,n,g,l,src in extra]
    assert len(rows)==27
    for i,row in enumerate(rows,1):
        row.update(id=f'T{i:02}', p=exact_mcnemar(row['gains'],row['losses']))
        row['bonferroni_p']=min(1.0,len(rows)*row['p'])
        row['reject_fwer_05']=row['bonferroni_p']<=0.05
    report=dict(policy='Retrospective reporting family, not preregistered; historical motivating runs excluded as exploratory',
        family_size=len(rows),threshold=0.05/len(rows),tests=rows,
        rl_endpoint=paired_interval(0,3,542),
        rl_within_repaired=paired_interval(1,4,542))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tests'},indent=2))


if __name__=='__main__':
    main()
