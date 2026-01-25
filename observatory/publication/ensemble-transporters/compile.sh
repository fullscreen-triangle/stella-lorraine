#!/bin/bash
# Compilation script for Ensemble Membrane Transporter Maxwell Demons paper

echo "Compiling Ensemble Membrane Transporter Maxwell Demons paper..."
echo "================================================================"

# Change to publication directory
cd "$(dirname "$0")"

# First pass: generate aux files
echo "First pass..."
pdflatex -interaction=nonstopmode ensemble-membrane-transporter-maxwell-demons.tex

# Generate bibliography
echo "Generating bibliography..."
bibtex ensemble-membrane-transporter-maxwell-demons

# Second pass: incorporate references
echo "Second pass..."
pdflatex -interaction=nonstopmode ensemble-membrane-transporter-maxwell-demons.tex

# Third pass: resolve all cross-references
echo "Third pass..."
pdflatex -interaction=nonstopmode ensemble-membrane-transporter-maxwell-demons.tex

echo ""
echo "================================================================"
echo "Compilation complete!"
echo "Output: ensemble-membrane-transporter-maxwell-demons.pdf"
echo "================================================================"
