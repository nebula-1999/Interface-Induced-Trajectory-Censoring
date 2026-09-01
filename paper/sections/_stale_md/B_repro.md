9. Reproducibility

```bash
# analysis and figures (no GPU)
pip install -r requirements.txt
python analysis/intent.py          # the single intent criterion, all four counts
python runs/final/final_table.py   # main / variance / error-rate tables (N/A enforced)
python figures/make_figs.py

# probe (needs a vLLM server)
python probe_react_full.py --model <path> --port 8000 --n 100 \
  --protocol {react,fc} --strength {optional,mandatory} \
  --fc-schema {terse,rich,strict} --parser-adapter cross_family \
  --temperature 0.0 --seed 0 --out traj.jsonl
python validate_arms.py            # per-arm admissibility: lines / rc / n_err / provenance / script hash
```

Model snapshots must be pinned: `tokenizer_config.json` inheritance is the mechanism
behind §5.1's false-positive capability check, so a family's behaviour can change with a
repository revision. Every model path, its HF revision, and the serving flags used are
recorded per arm in the trajectory provenance and in `runs/final/by_config/README.md`.

Trajectories for all 50 full-length arms, the training logs, the errata, and the
per-configuration index are in `runs/final/`. Third-party parser: hanXen
(`1b92150`, Apache 2.0), few-shot examples rewritten from `get_weather` to `run_tests`;
original retained with both hashes recorded.
