@echo off
REM Compile the observation boundary paper (Windows)

echo Compiling observation boundary paper...

REM Run pdflatex first time (generates .aux file)
pdflatex -interaction=nonstopmode properties-of-observation-boundary.tex

REM Run bibtex (processes citations)
bibtex properties-of-observation-boundary

REM Run pdflatex twice more (resolves references)
pdflatex -interaction=nonstopmode properties-of-observation-boundary.tex
pdflatex -interaction=nonstopmode properties-of-observation-boundary.tex

echo Compilation complete! Output: properties-of-observation-boundary.pdf

REM Clean up auxiliary files (optional)
REM del *.aux *.log *.out *.toc *.bbl *.blg

pause
