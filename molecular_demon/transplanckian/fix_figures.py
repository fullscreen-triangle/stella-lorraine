"""
Fix LaTeX figure placement issues by converting [htbp] to [H] for non-floating figures
"""

import re

# Read the figures.tex file
with open('figures.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace [htbp] with [H] for all figures to force exact placement
content_fixed = content.replace('\\begin{figure}[htbp]', '\\begin{figure}[H]')

# Write back
with open('figures_fixed.tex', 'w', encoding='utf-8') as f:
    f.write(content_fixed)

print("[OK] Created figures_fixed.tex with [H] placement specifiers")
print("To use: replace \\input{figures} with \\input{figures_fixed} in your main .tex file")
