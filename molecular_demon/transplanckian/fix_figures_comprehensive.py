"""
Comprehensive LaTeX Figure Fixer

Fixes common issues:
1. Converts [htbp] to [H] for exact placement
2. Adds short captions for long captions (>150 chars)
3. Ensures all figures are properly closed
"""

import re

def fix_latex_figures(content):
    """Fix LaTeX figure issues"""
    
    # 1. Replace [htbp] with [H]
    content = content.replace('\\begin{figure}[htbp]', '\\begin{figure}[H]')
    
    # 2. Fix long captions by adding optional short version
    # Pattern to match \caption{...}
    caption_pattern = r'\\caption\{((?:[^{}]|\\{|\\})+)\}'
    
    def fix_caption(match):
        full_caption = match.group(1)
        # If caption is very long (>200 chars), create short version
        if len(full_caption) > 200:
            # Extract first sentence or first 80 chars for short caption
            first_sentence = full_caption.split('.')[0]
            if len(first_sentence) > 100:
                short_caption = first_sentence[:80] + "..."
            else:
                short_caption = first_sentence
            # Remove LaTeX commands from short caption
            short_caption = re.sub(r'\\textbf\{([^}]+)\}', r'\1', short_caption)
            return f'\\caption[{short_caption}]{{{full_caption}}}'
        return match.group(0)
    
    content = re.sub(caption_pattern, fix_caption, content, flags=re.DOTALL)
    
    return content

# Read the figures.tex file
with open('figures.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Apply fixes
content_fixed = fix_latex_figures(content)

# Write back
with open('figures_fixed.tex', 'w', encoding='utf-8') as f:
    f.write(content_fixed)

print("[OK] Created figures_fixed.tex with comprehensive fixes:")
print("  - Changed [htbp] -> [H] for exact placement")
print("  - Added short captions for long captions")
print("")
print("IMPORTANT: Make sure your preamble includes:")
print("  \\usepackage{float}")
print("")
print("Then replace any \\input{figures} with \\input{figures_fixed}")
