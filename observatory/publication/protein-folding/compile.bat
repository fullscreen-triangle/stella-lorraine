@echo off
REM Compilation script for GroEL Phase-Locking Resonance Chamber paper (Windows)

echo Compiling GroEL Phase-Locking Resonance Chamber paper...
echo ==========================================================

REM Change to publication directory
cd /d "%~dp0"

REM First pass: generate aux files
echo First pass...
pdflatex -interaction=nonstopmode groel-phase-locking-resonance-chamber.tex

REM Generate bibliography
echo Generating bibliography...
bibtex groel-phase-locking-resonance-chamber

REM Second pass: incorporate references
echo Second pass...
pdflatex -interaction=nonstopmode groel-phase-locking-resonance-chamber.tex

REM Third pass: resolve all cross-references
echo Third pass...
pdflatex -interaction=nonstopmode groel-phase-locking-resonance-chamber.tex

echo.
echo ==========================================================
echo Compilation complete!
echo Output: groel-phase-locking-resonance-chamber.pdf
echo ==========================================================
pause
