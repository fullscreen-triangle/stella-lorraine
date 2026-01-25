@echo off
REM LaTeX compilation script with bibliography
REM Proper workflow: pdflatex -> bibtex -> pdflatex -> pdflatex

echo ======================================================================
echo Compiling hardware-based-temporal-measurements.tex
echo ======================================================================

echo.
echo [1/4] First pdflatex pass (generating .aux file)...
pdflatex -interaction=nonstopmode hardware-based-temporal-measurements.tex
if %errorlevel% neq 0 (
    echo ERROR: First pdflatex pass failed!
    exit /b 1
)

echo.
echo [2/4] Running bibtex (processing citations)...
bibtex hardware-based-temporal-measurements
if %errorlevel% neq 0 (
    echo WARNING: bibtex reported errors, but continuing...
)

echo.
echo [3/4] Second pdflatex pass (including bibliography)...
pdflatex -interaction=nonstopmode hardware-based-temporal-measurements.tex
if %errorlevel% neq 0 (
    echo ERROR: Second pdflatex pass failed!
    exit /b 1
)

echo.
echo [4/4] Third pdflatex pass (resolving all references)...
pdflatex -interaction=nonstopmode hardware-based-temporal-measurements.tex
if %errorlevel% neq 0 (
    echo ERROR: Third pdflatex pass failed!
    exit /b 1
)

echo.
echo ======================================================================
echo Compilation complete! Output: hardware-based-temporal-measurements.pdf
echo ======================================================================

REM Optional: Clean up auxiliary files
REM del *.aux *.log *.out *.bbl *.blg *.toc
