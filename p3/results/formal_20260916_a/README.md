# Formal P3 evidence bundle

This directory preserves the evidence retrieved on 2026-09-19, after the formal
single-seed broken/repaired comparison completed. No additional training was run
for this audit. See `p3/FINAL_EVIDENCE_AUDIT_20260919.md` for interpretation.

## Reproduce locally

From the repository root:

```bash
shasum -a 256 p3/results/formal_20260916_a/p3_evidence_20260919.tar.gz
mkdir -p p3/results/formal_20260916_a/raw
tar -xzf p3/results/formal_20260916_a/p3_evidence_20260919.tar.gz \
  -C p3/results/formal_20260916_a/raw
python3 analysis/p3_final_evidence.py p3/results/formal_20260916_a/raw \
  --output /tmp/p3-reproduced-audit.json
cmp /tmp/p3-reproduced-audit.json p3/results/formal_20260916_a/audit.json
```

Expected archive SHA-256:
`bfb0aa8447292726d09eac1d140cbb031635d12e100fd2a18a42ea7208e5c1cc`.
The extracted directory is ignored by Git to avoid storing the same records twice.
It contains full evaluation rows, rollout events, logs, provenance, and deployed
source. Original repaired steps 91–116 are retained but excluded from final-path
totals; resumed steps 91–150 replace them.

The heavy model/optimizer checkpoints are not included. Their in-place
server-side comparison is recorded in `checkpoint_lora_audit.json` and can be
repeated with `analysis/p3_checkpoint_lora_audit.py` if the retained checkpoints
are available. This local bundle alone cannot reproduce that tensor comparison.

The bundle is public research provenance, **not anonymized conference
supplementary material**: paths and source metadata may identify the project.
