#!/usr/bin/env python3
"""
Setup SMARTS Directory
======================
Creates the smarts directory and provides instructions for adding SMARTS files.
"""

import os

def setup_smarts_directory():
    """Create smarts directory and provide instructions"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    smarts_dir = os.path.join(current_dir, 'smarts')

    # Create directory if it doesn't exist
    os.makedirs(smarts_dir, exist_ok=True)

    print("\n" + "="*70)
    print("   SMARTS DIRECTORY SETUP")
    print("="*70)

    print(f"\n📁 SMARTS directory created/verified:")
    print(f"   {smarts_dir}")

    # Check for existing files
    expected_files = ['agrafiotis.smarts', 'ahmed.smarts', 'hann.smarts']
    found_files = []
    missing_files = []

    for filename in expected_files:
        filepath = os.path.join(smarts_dir, filename)
        if os.path.exists(filepath):
            # Count lines
            with open(filepath, 'r') as f:
                lines = sum(1 for line in f if line.strip() and not line.startswith('#'))
            found_files.append((filename, lines))
        else:
            missing_files.append(filename)

    if found_files:
        print(f"\n✅ Found SMARTS files:")
        for filename, lines in found_files:
            print(f"   - {filename}: {lines} patterns")

    if missing_files:
        print(f"\n⚠️  Missing SMARTS files:")
        for filename in missing_files:
            print(f"   - {filename}")

        print(f"\n📋 To add SMARTS files:")
        print(f"   1. Place your .smarts files in: {smarts_dir}")
        print(f"   2. Expected files: {', '.join(expected_files)}")
        print(f"   3. Format: One SMARTS pattern per line")
        print(f"   4. Comments start with '#'")

        # Create example file
        example_file = os.path.join(smarts_dir, 'example.smarts')
        if not os.path.exists(example_file):
            with open(example_file, 'w') as f:
                f.write("# Example SMARTS patterns\n")
                f.write("c1ccccc1  # benzene\n")
                f.write("CCO  # ethanol\n")
                f.write("CC(=O)O  # acetic acid\n")
                f.write("c1ccc2ccccc2c1  # naphthalene\n")
            print(f"\n   ✓ Created example file: example.smarts")
    else:
        print(f"\n✅ All SMARTS files found!")
        print(f"   Total patterns: {sum(lines for _, lines in found_files)}")

    print(f"\n{'='*70}")
    print(f"   Run 'python led_excitation.py' to test with your SMARTS files")
    print(f"{'='*70}\n")

    return smarts_dir


if __name__ == "__main__":
    setup_smarts_directory()
