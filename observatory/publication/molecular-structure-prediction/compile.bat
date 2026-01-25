@echo off
REM Compilation script for Molecular Structure Prediction paper (Windows)

echo Compiling Molecular Structure Prediction paper...
echo ==================================================

REM Change to publication directory
cd /d "%~dp0"

REM First pass: generate aux files
echo First pass...
pdflatex -interaction=nonstopmode molecular-structure-prediction.tex

REM Generate bibliography
echo Generating bibliography...
bibtex molecular-structure-prediction

REM Second pass: incorporate references
echo Second pass...
pdflatex -interaction=nonstopmode molecular-structure-prediction.tex

REM Third pass: resolve all cross-references
echo Third pass...
pdflatex -interaction=nonstopmode molecular-structure-prediction.tex

echo.
echo ==================================================
echo Compilation complete!
echo Output: molecular-structure-prediction.pdf
echo ==================================================
pause
