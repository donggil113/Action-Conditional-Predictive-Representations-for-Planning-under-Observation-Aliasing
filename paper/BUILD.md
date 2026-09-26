# Manuscript build

## Format source (official, unmodified)
- Author Guidelines: https://iclr.cc/Conferences/2027/AuthorGuidelines (checked 2026-09-26). The page links the style ZIP below. Main text limit: 9 pages at submission. An AI use statement is required and does not count toward the limit; a reproducibility statement is optional.
- ZIP: https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip. A copy is kept at `style_provenance/iclr-2027-style-files.zip`.
  - sha256 `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`, 39,348 bytes. The ZIP entries are dated 2026-07-28 for `.sty` and `.tex` and 2025-06-25 for the other files.
- Files copied **unmodified** from the ZIP into `paper/`:

| file | sha256 |
|---|---|
| iclr2027_conference.sty | 797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6 |
| iclr2027_conference.bst | 2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5 |
| fancyhdr.sty | b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea |
| natbib.sty | 88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d |
| math_commands.tex | 90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9 |

### Document-level deviations (the style files themselves are untouched)
- In anonymous mode the style prints the header "Under review as a conference paper at ICLR 2027". This draft has **not been submitted**, so `main.tex` replaces the running-header text after `\maketitle` with `\lhead{Internal working draft --- not submitted}`. The title also carries a line "Internal working draft v0 --- not submitted to any venue".
- The anonymous author block "Anonymous authors / Paper under double-blind review" is hard-coded in the style's `\@maketitle`. It is left as produced and is **not** a status claim. No submission ID, `\iclrfinalcopy`, acceptance status, or anonymous URL is used.
- Added packages: `booktabs`, `graphicx`, `amsmath`, `amssymb`. No font-size, margin, or spacing changes.

## Toolchain (installed for this build)
No LaTeX engine was present. GPG-verified Ubuntu (noble) archive packages were installed with `apt-get install --no-install-recommends`, 18 packages in total. The main ones:
- texlive-base, texlive-latex-base, texlive-latex-recommended, texlive-fonts-recommended: 2023.20240207-1
- texlive-binaries: 2023.20230311.66589-9build3, giving pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian) and BibTeX
- poppler-utils 24.02.0-1ubuntu9.9, used only to render PDF pages for visual checking

The full list is in `style_provenance/texlive_packages_installed.tsv`.

## Build
```bash
paper/build.sh
```
The script runs, in order:
1. `python3 export_results.py`
2. pdflatex
3. bibtex
4. pdflatex
5. pdflatex

It then prints the total page count and the page on which the main text ends (from the `sec:end-of-main-text` label).

## Build CPU ledger (cap 600 CPU-s, separate from experiments)
| step | CPU (user+sys) |
|---|---|
| apt-get update | 3.2 s |
| install TeX Live packages | 35.6 s |
| install poppler-utils | 2.2 s |
| v0 build (`build.sh`, export + 3×pdflatex + bibtex) | 1.2 s |
| exporter test runs | 0.1 s |
| **v0 total** | **≈42 s** |

Later builds are appended in PAPER_STATUS.md.
