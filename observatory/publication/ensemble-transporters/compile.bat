@echo off
REM Compilation script for Ensemble Membrane Transporter Maxwell Demons paper (Windows)

echo Compiling Ensemble Membrane Transporter Maxwell Demons paper...
echo ================================================================

REM Change to publication directory
cd /d "%~dp0"

REM First pass: generate aux files
echo First pass...
pdflatex -interaction=nonstopmode ensemble-membrane-transporter-maxwell-demons.tex

REM Generate bibliography
echo Generating bibliography...
bibtex ensemble-membrane-transporter-maxwell-demons

REM Second pass: incorporate references
echo Second pass...
pdflatex -interaction=nonstopmode ensemble-membrane-transporter-maxwell-demons.tex

REM Third pass: resolve all cross-references
echo Third pass...
pdflatex -interaction=nonstopmode ensemble-membrane-transporter-maxwell-demons.tex

echo.
echo ================================================================
echo Compilation complete!
echo Output: ensemble-membrane-transporter-maxwell-demons.pdf
echo ================================================================
pause
