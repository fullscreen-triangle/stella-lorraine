#!/bin/bash
# LaTeX compilation script with bibliography
# Proper workflow: pdflatex -> bibtex -> pdflatex -> pdflatex

echo "======================================================================"
echo "Compiling hardware-based-temporal-measurements.tex"
echo "======================================================================"

echo ""
echo "[1/4] First pdflatex pass (generating .aux file)..."
pdflatex -interaction=nonstopmode hardware-based-temporal-measurements.tex
if [ $? -ne 0 ]; then
    echo "ERROR: First pdflatex pass failed!"
    exit 1
fi

echo ""
echo "[2/4] Running bibtex (processing citations)..."
bibtex hardware-based-temporal-measurements
if [ $? -ne 0 ]; then
    echo "WARNING: bibtex reported errors, but continuing..."
fi

echo ""
echo "[3/4] Second pdflatex pass (including bibliography)..."
pdflatex -interaction=nonstopmode hardware-based-temporal-measurements.tex
if [ $? -ne 0 ]; then
    echo "ERROR: Second pdflatex pass failed!"
    exit 1
fi

echo ""
echo "[4/4] Third pdflatex pass (resolving all references)..."
pdflatex -interaction=nonstopmode hardware-based-temporal-measurements.tex
if [ $? -ne 0 ]; then
    echo "ERROR: Third pdflatex pass failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Compilation complete! Output: hardware-based-temporal-measurements.pdf"
echo "======================================================================"

# Optional: Clean up auxiliary files
# rm -f *.aux *.log *.out *.bbl *.blg *.toc
