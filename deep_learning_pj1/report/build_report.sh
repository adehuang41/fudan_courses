#!/bin/zsh
set -e
cd "$(dirname "$0")"
latexmk -xelatex -interaction=nonstopmode report.tex
