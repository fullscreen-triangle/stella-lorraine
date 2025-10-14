#!/usr/bin/env python3
"""Quick GPS Data Inspector - Shows basic info about your messy GPS dataset"""

import json
import sys
import os

# Add path
sys.path.insert(0, os.path.dirname(__file__))

print("="*70)
print("   QUICK GPS DATA INSPECTOR")
print("="*70)

# Load data
file_path = 'gps_dataset.json'
print(f"\n📍 Loading: {file_path}")

try:
    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded successfully")
    print(f"\nData type: {data.get('type', 'Unknown')}")

    features = data.get('features', [])
    print(f"Total features: {len(features)}")

    # Count feature types
    points = 0
    linestrings = 0
    linestring_sizes = []

    for feature in features:
        geom_type = feature.get('geometry', {}).get('type', 'Unknown')
        if geom_type == 'Point':
            points += 1
        elif geom_type == 'LineString':
            coords = feature.get('geometry', {}).get('coordinates', [])
            linestrings += 1
            linestring_sizes.append(len(coords))

    print(f"\n📊 Feature breakdown:")
    print(f"   Points: {points}")
    print(f"   LineStrings (tracks): {linestrings}")

    if linestrings > 0:
        print(f"\n🏃 Track details:")
        for i, size in enumerate(sorted(linestring_sizes, reverse=True)):
            print(f"   Track {i+1}: {size} GPS points")

        print(f"\n✨ Found {linestrings} tracks!")

        if linestrings >= 2:
            print(f"\n   The two largest tracks are likely your two watches:")
            print(f"   - Watch 1: {sorted(linestring_sizes, reverse=True)[0]} points")
            print(f"   - Watch 2: {sorted(linestring_sizes, reverse=True)[1]} points")

        # Show sample coordinates from largest track
        for feature in features:
            if feature.get('geometry', {}).get('type') == 'LineString':
                coords = feature.get('geometry', {}).get('coordinates', [])
                if len(coords) == max(linestring_sizes):
                    print(f"\n📍 Sample coordinates from largest track:")
                    print(f"   Start: {coords[0]}")
                    print(f"   End: {coords[-1]}")
                    break

except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
    print(f"   Make sure you're running this from: observatory/src/precision/")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("To extract and analyze the tracks, run:")
print("python analyze_messy_gps.py")
print("="*70)
