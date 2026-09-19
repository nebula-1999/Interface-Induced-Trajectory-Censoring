# Public preprint update — 2026-09-19

Upload `arxiv_submission.tar.gz` as the source package for the existing preprint
`2609.03966`. This prepares a replacement version; it does not submit it.

## Author metadata (in order)

1. Wenbo Wang — City University of Hong Kong — wenbwang3-c@my.cityu.edu.hk
2. Di Sang — Beijing Institute of Technology — 3120265429@bit.edu.cn

Enter both authors in the submission metadata as well as retaining them in the
manuscript. The public PDF metadata lists both authors. The ICLR manuscript
remains anonymous and is not included in this package.

## Files and build

- Main source: `main.tex`; compiler: **XeLaTeX**.
- The archive includes `main.bbl`, every referenced section, and the five figures.
- Chinese verbatim research evidence remains in the appendix. Fonts use TeX
  Live's bundled Fandol files, not machine-specific macOS fonts.
- Paste `abstract_arxiv.txt` into the abstract field: ASCII-only and under
  1,920 characters. It incorporates the corrected BFCL interpretation and the
  completed formal P3 comparison; do not reuse the previous abstract.
- `main.pdf` is the locally verified preview, not included in the source archive.

After upload, review arXiv's compiled preview, author order, affiliations, abstract,
and bibliography before finalizing. Local compilation cannot guarantee acceptance
by the remote compiler or moderation system.

## Publication boundary

This update contains the public long manuscript and its build assets only. It
does not upload the P3 raw rollout/log archive or the anonymous ICLR version.
