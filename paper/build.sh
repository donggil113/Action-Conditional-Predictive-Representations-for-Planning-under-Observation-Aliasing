#!/usr/bin/env bash
# Build the working draft with the locally installed TeX Live (see BUILD.md).
set -euo pipefail
cd "$(dirname "$0")"
python3 export_results.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build_pass1.log
bibtex main > build_bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build_pass2.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build_pass3.log
grep -E "Output written|LaTeX Warning|undefined" main.log | sort | uniq -c || true
python3 - <<'PY'
import re
aux = open("main.aux").read()
m = re.search(r"\\newlabel\{sec:end-of-main-text\}\{\{[^}]*\}\{(\d+)\}", aux)
log = open("main.log").read()
pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
print("total_pages", pages.group(1) if pages else "?", "| main_text_ends_on_page", m.group(1) if m else "?")
PY
