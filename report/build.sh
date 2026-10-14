#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python3 fetch_style.py
mkdir -p build
TEXINPUTS=".style:${TEXINPUTS:-}" pdflatex -halt-on-error -interaction=nonstopmode -output-directory=build report.tex
TEXINPUTS=".style:${TEXINPUTS:-}" pdflatex -halt-on-error -interaction=nonstopmode -output-directory=build report.tex
cp build/report.pdf report.pdf
