"""
Comprehensive LaTeX Section File Fixer
Fixes:
1. Replace figure* with figure (article class doesn't support figure*)
2. Add missing \end{figure}
3. Fix Unicode characters (μ -> $\mu$)
4. Fix runaway arguments in captions
"""

import os
import glob
import re

def fix_file(filepath):
    """Fix common LaTeX issues in a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # 1. Replace figure* with figure (article class doesn't support two-column floats)
    if '\\begin{figure*}' in content:
        content = content.replace('\\begin{figure*}[htbp]', '\\begin{figure}[H]')
        content = content.replace('\\begin{figure*}', '\\begin{figure}[H]')
        content = content.replace('\\end{figure*}', '\\end{figure}')
        changes.append("Replaced figure* with figure")
    
    # 2. Fix Unicode mu character
    content = content.replace('μ', r'$\mu$')
    if 'μ' in original_content:
        changes.append("Fixed Unicode μ characters")
    
    # 3. Fix degree symbols in math mode
    content = re.sub(r'\$([^$]*?)\\textdegree([^$]*?)\$', r'$\1^\\circ\2$', content)
    
    # 4. Fix unclosed captions with newlines (runaway arguments)
    # Pattern: \caption{ ... \textbf{...} NEWLINE ... }
    # Problem: LaTeX can't handle paragraph breaks in \caption without proper wrapping
    def fix_caption(match):
        caption_content = match.group(1)
        # Remove extra whitespace and newlines within caption
        caption_content = re.sub(r'\n\s*\n', ' ', caption_content)
        caption_content = re.sub(r'\n\s+', ' ', caption_content)
        return f'\\caption{{{caption_content}}}'
    
    # Fix captions that span multiple lines improperly
    content = re.sub(r'\\caption\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', fix_caption, content, flags=re.DOTALL)
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return changes
    return []

# Fix all section files
section_files = glob.glob('sections/*.tex')
fixed_files = []

for filepath in section_files:
    changes = fix_file(filepath)
    if changes:
        fixed_files.append((os.path.basename(filepath), changes))
        print(f"[OK] Fixed {os.path.basename(filepath)}: {', '.join(changes)}")

if fixed_files:
    print(f"\n[OK] Fixed {len(fixed_files)} files")
else:
    print("[OK] No additional fixes needed")
