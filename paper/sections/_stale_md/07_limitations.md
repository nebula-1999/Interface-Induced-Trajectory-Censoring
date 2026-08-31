7. Limitations

1. **Sample.** One task family (decontaminated KodCode), 100 items per arm, and the items
   are `clean[:100]` — **not a random sample**. Pairing is exact but generalisation to the
   corpus is unestablished.
2. **Single-sample main table.** §5.4 is `temperature=0`, one sample. Variance is estimated
   only for Llama-3.1-8B, and only 2 of 3 FC arms are admissible.
3. **Multiple comparisons.** Roughly a dozen McNemar tests, **uncorrected**. The main
   effects (~1e-3) survive Bonferroni; marginal results (*p*=0.093, 0.125) do not.
4. **Adapter attribution.** §5.5 changes parser, template and few-shot together; the
   2×2 ablation isolating the parser was not run.
5. **Scale.** Up to 32B; no frontier models. Single A800.
6. **Training evidence is a mechanism demonstration** (10 and 3 steps), not a 150-step
   outcome comparison. We claim gradient-unreachability, not a quantified final gap.
7. **The residual protocol effect rests on one family.** Conditioned on successful parsing
   it holds on Llama-3.1-8B (*p*=0.0021) but not on Qwen2.5-Coder-7B (*p*=0.119), and
   Mistral admits no comparison. Claims about "ReAct beats function calling" should be
   read as family-specific.
8. **Instrumentation errata.** Seven ReAct arms recorded `max_tokens=2048` while actually
   running at 1024 (an asymmetry that favoured FC; re-running changed final pass by ≤1
   item); Llama's initiation rate required manual correction from 97 to 74; five arms are
   error-bearing and admissible only for error-rate census. All are enumerated in
   `ERRATA.md` and none is silently absorbed into a reported number.

---
