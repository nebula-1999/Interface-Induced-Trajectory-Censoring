# ICLR 2027 submission version

This directory contains an anonymous, English-only submission manuscript built with the official
ICLR 2027 style. It is intentionally separate from `paper/main.tex`, which remains the longer arXiv
version.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The checked PDF is `main.pdf` (US Letter).
`abstract_openreview.txt` is a plain-text, ASCII-safe copy for the submission form and remains
below the 1,920-character field limit.

## Change log

- Rebuilt the paper around the measurement-validity claim rather than a catalogue of stack bugs.
- Led with the BFCL v4 external result and retained its template-by-parser complementarity result.
- Kept all headline experiments in the main paper: BFCL, tau-bench, the mismatched and matched
  scale ladders, the within-lineage Instruct control, Llama strict decoding, evaluation repair, and
  the formal 7B broken-versus-repaired RL control.
- Added a compact framework with testable downstream signatures for parser-layer masking versus
  generation-time suppression.
- Separated mechanism outcomes from task outcomes and stated the exact McNemar/Bonferroni policy.
- Added the required ICLR 2027 AI-use statement and a reproducibility statement.
- Preserved the complete Data Errata and all 17 limitations after the references.
- Replaced the Chinese prompt appendix with English renderings; byte-exact originals remain in the
  anonymous artifact and the arXiv source.
- Left the long arXiv manuscript and all of its source sections intact.

## Page accounting

The 2026-09-21 verified build has 30 pages including references and appendices;
the main text ends on page 8. The page-limit accounting is:

| material | pages | counted toward 9-page limit |
|---|---:|---:|
| Introduction | 1--2 | yes |
| Related work + framework | 2--3 | yes |
| Measurement setup | 3 | yes |
| Results | 4--7 | yes |
| Implications, limitations, conclusion | 7--8 | yes |
| AI-use + reproducibility statements | 8 | no (per ICLR 2027 template) |
| References | 9--10 | no |
| Appendices (including retrospective statistical inventory) | 10--30 | no |

The main text therefore ends on **page 8**, leaving one page of safety margin.
The added P3 evidence and corrected statistical/methodological descriptions
account for the extra main-text page relative to the earlier seven-page build.

## Material moved out of the main text

No result was deleted from the project. The following are summarized in the main paper and fully
tabulated after the references: annotation rounds and adjudication, offline replay matrices,
failure-layer tables, arm inventory, prompts and schemas, exact serving/training configuration,
pressure controls, sampling variance, Jaccard/composition analysis, historical RL diagnostics,
resume provenance, and all instrumentation errata. Mistral and DeepSeek remain explicitly disclosed
as asymmetric evidence rather than presented as comparable quantitative arms.
