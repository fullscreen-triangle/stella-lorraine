"""
Fix all section files by replacing [htbp] with [H] for figure placement
"""

import os
import glob

# Get all .tex files in sections directory
section_files = glob.glob('sections/*.tex')

fixed_count = 0
for filepath in section_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace [htbp] with [H]
    if '[htbp]' in content:
        content_fixed = content.replace('\\begin{figure}[htbp]', '\\begin{figure}[H]')
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content_fixed)
        
        fixed_count += 1
        print(f"[OK] Fixed {os.path.basename(filepath)}")

print(f"\n[OK] Fixed {fixed_count} section files")
print("All figures now use [H] placement specifier")
