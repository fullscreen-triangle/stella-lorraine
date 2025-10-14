#!/usr/bin/env python3
"""
GPS Data Converter
==================
Converts various smartwatch GPS formats (GPX, TCX, FIT, JSON) to CSV
for use with GPS precision analysis.
"""

import xml.etree.ElementTree as ET
import pandas as pd
import json
import os
import sys
from datetime import datetime

def convert_gpx_to_csv(gpx_file):
    """Convert GPX format to CSV"""
    print(f"📍 Converting GPX: {gpx_file}")

    tree = ET.parse(gpx_file)
    root = tree.getroot()

    # Handle namespace
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

    data = []

    # Find all track points
    for trkpt in root.findall('.//gpx:trkpt', ns):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))

        # Extract other data
        time_elem = trkpt.find('gpx:time', ns)
        ele_elem = trkpt.find('gpx:ele', ns)

        point = {
            'latitude': lat,
            'longitude': lon
        }

        if time_elem is not None:
            point['timestamp'] = time_elem.text

        if ele_elem is not None:
            point['altitude'] = float(ele_elem.text)

        # Try to find speed (extension)
        for ext in trkpt.findall('.//gpx:speed', ns):
            point['speed'] = float(ext.text)

        data.append(point)

    df = pd.DataFrame(data)
    print(f"   ✓ Extracted {len(df)} points")
    return df

def convert_tcx_to_csv(tcx_file):
    """Convert TCX format to CSV"""
    print(f"📍 Converting TCX: {tcx_file}")

    tree = ET.parse(tcx_file)
    root = tree.getroot()

    # TCX namespace
    ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

    data = []

    for trackpoint in root.findall('.//tcx:Trackpoint', ns):
        point = {}

        time_elem = trackpoint.find('tcx:Time', ns)
        if time_elem is not None:
            point['timestamp'] = time_elem.text

        pos = trackpoint.find('tcx:Position', ns)
        if pos is not None:
            lat_elem = pos.find('tcx:LatitudeDegrees', ns)
            lon_elem = pos.find('tcx:LongitudeDegrees', ns)

            if lat_elem is not None:
                point['latitude'] = float(lat_elem.text)
            if lon_elem is not None:
                point['longitude'] = float(lon_elem.text)

        alt_elem = trackpoint.find('tcx:AltitudeMeters', ns)
        if alt_elem is not None:
            point['altitude'] = float(alt_elem.text)

        # Speed might be in extensions
        for ext in trackpoint.findall('.//tcx:Speed', ns):
            point['speed'] = float(ext.text)

        if 'latitude' in point and 'longitude' in point:
            data.append(point)

    df = pd.DataFrame(data)
    print(f"   ✓ Extracted {len(df)} points")
    return df

def convert_json_to_csv(json_file):
    """Convert JSON format to CSV"""
    print(f"📍 Converting JSON: {json_file}")

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        if 'locations' in data:
            df = pd.DataFrame(data['locations'])
        elif 'points' in data:
            df = pd.DataFrame(data['points'])
        else:
            df = pd.DataFrame([data])

    # Standardize column names
    column_mapping = {
        'lat': 'latitude',
        'lng': 'longitude',
        'lon': 'longitude',
        'time': 'timestamp',
        'timestampMs': 'timestamp',
        'velocity': 'speed',
        'elevation': 'altitude',
        'alt': 'altitude'
    }

    df = df.rename(columns=column_mapping)
    print(f"   ✓ Extracted {len(df)} points")
    return df

def calculate_speed_from_positions(df):
    """Calculate speed if not provided"""
    if 'speed' not in df.columns:
        print("   ℹ Calculating speed from position changes...")

        speeds = [0]  # First point has no speed

        for i in range(1, len(df)):
            # Haversine distance
            lat1, lon1 = df.iloc[i-1][['latitude', 'longitude']]
            lat2, lon2 = df.iloc[i][['latitude', 'longitude']]

            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)

            a = (np.sin(dlat/2)**2 +
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
                 np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6371000 * c  # Earth radius in meters

            # Time difference
            if 'timestamp' in df.columns:
                t1 = pd.to_datetime(df.iloc[i-1]['timestamp'])
                t2 = pd.to_datetime(df.iloc[i]['timestamp'])
                dt = (t2 - t1).total_seconds()
            else:
                dt = 1.0  # Assume 1 second

            speed = distance / dt if dt > 0 else 0
            speeds.append(speed)

        df['speed'] = speeds
        print(f"   ✓ Speed calculated (mean: {np.mean(speeds):.2f} m/s)")

    return df

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Convert smartwatch GPS data to CSV')
    parser.add_argument('--input', '-i', required=True, help='Input file (GPX, TCX, JSON)')
    parser.add_argument('--output', '-o', help='Output CSV file (default: same name with .csv)')

    args = parser.parse_args()

    input_file = args.input

    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        sys.exit(1)

    # Determine format
    ext = os.path.splitext(input_file)[1].lower()

    print("="*70)
    print("   GPS DATA CONVERTER")
    print("="*70)

    # Convert based on format
    if ext == '.gpx':
        df = convert_gpx_to_csv(input_file)
    elif ext == '.tcx':
        df = convert_tcx_to_csv(input_file)
    elif ext == '.json':
        df = convert_json_to_csv(input_file)
    else:
        print(f"❌ Unsupported format: {ext}")
        print("   Supported: .gpx, .tcx, .json")
        sys.exit(1)

    # Calculate speed if missing
    import numpy as np
    df = calculate_speed_from_positions(df)

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.splitext(input_file)[0] + '.csv'

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"\n✓ Conversion complete!")
    print(f"   Input:  {input_file}")
    print(f"   Output: {output_file}")
    print(f"   Points: {len(df)}")

    if 'timestamp' in df.columns:
        print(f"   Time:   {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    if 'speed' in df.columns:
        print(f"   Speed:  {df['speed'].mean():.2f} ± {df['speed'].std():.2f} m/s")

    print(f"\n   Ready for analysis:")
    print(f"   python gps_precision_analysis.py --watch1 {output_file}")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
