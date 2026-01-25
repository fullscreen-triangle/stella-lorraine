#!/bin/bash
# Compilation script for Molecular Structure Prediction paper

echo "Compiling Molecular Structure Prediction paper..."
echo "=================================================="

# Change to publication directory
cd "$(dirname "$0")"

# First pass: generate aux files
echo "First pass..."
pdflatex -interaction=nonstopmode molecular-structure-prediction.tex

# Generate bibliography
echo "Generating bibliography..."
bibtex molecular-structure-prediction

# Second pass: incorporate references
echo "Second pass..."
pdflatex -interaction=nonstopmode molecular-structure-prediction.tex

# Third pass: resolve all cross-references
echo "Third pass..."
pdflatex -interaction=nonstopmode molecular-structure-prediction.tex

echo ""
echo "=================================================="
echo "Compilation complete!"
echo "Output: molecular-structure-prediction.pdf"
echo "=================================================="
