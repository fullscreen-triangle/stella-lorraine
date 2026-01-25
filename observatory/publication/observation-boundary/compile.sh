#!/bin/bash

# Compile the observation boundary paper

echo "Compiling observation boundary paper..."

# Run pdflatex first time (generates .aux file)
pdflatex -interaction=nonstopmode properties-of-observation-boundary.tex

# Run bibtex (processes citations)
bibtex properties-of-observation-boundary

# Run pdflatex twice more (resolves references)
pdflatex -interaction=nonstopmode properties-of-observation-boundary.tex
pdflatex -interaction=nonstopmode properties-of-observation-boundary.tex

echo "Compilation complete! Output: properties-of-observation-boundary.pdf"

# Clean up auxiliary files (optional)
# rm -f *.aux *.log *.out *.toc *.bbl *.blg
