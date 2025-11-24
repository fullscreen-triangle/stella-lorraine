#!/bin/bash
# Compilation script for GroEL Phase-Locking Resonance Chamber paper

echo "Compiling GroEL Phase-Locking Resonance Chamber paper..."
echo "=========================================================="

# Change to publication directory
cd "$(dirname "$0")"

# First pass: generate aux files
echo "First pass..."
pdflatex -interaction=nonstopmode groel-phase-locking-resonance-chamber.tex

# Generate bibliography
echo "Generating bibliography..."
bibtex groel-phase-locking-resonance-chamber

# Second pass: incorporate references
echo "Second pass..."
pdflatex -interaction=nonstopmode groel-phase-locking-resonance-chamber.tex

# Third pass: resolve all cross-references
echo "Third pass..."
pdflatex -interaction=nonstopmode groel-phase-locking-resonance-chamber.tex

# Clean up auxiliary files (optional - uncomment if desired)
# rm -f *.aux *.log *.out *.toc *.bbl *.blg

echo ""
echo "=========================================================="
echo "Compilation complete!"
echo "Output: groel-phase-locking-resonance-chamber.pdf"
echo "=========================================================="
