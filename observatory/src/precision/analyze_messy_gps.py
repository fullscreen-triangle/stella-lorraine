#!/usr/bin/env python3
"""
Messy GPS Data Analyzer
========================
Analyzes and separates GPS tracks from mixed/messy GeoJSON data.
Identifies different watches, cleans data, and prepares for precision analysis.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import os

def load_and_analyze_geojson(file_path):
    """Load GeoJSON and identify different tracks"""
    print("="*70)
    print("   MESSY GPS DATA ANALYZER")
    print("="*70)

    print(f"\n📍 Loading: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"   Type: {data.get('type', 'Unknown')}")

    if data['type'] != 'FeatureCollection':
        print("   ⚠ Not a FeatureCollection, trying to parse anyway...")

    features = data.get('features', [])
    print(f"   Total features: {len(features)}")

    # Categorize features
    points = []
    linestrings = []
    other = []

    for feature in features:
        geometry = feature.get('geometry')
        if geometry is None:
            other.append(feature)
            continue

        geom_type = geometry.get('type', 'Unknown')
        if geom_type == 'Point':
            points.append(feature)
        elif geom_type == 'LineString':
            linestrings.append(feature)
        else:
            other.append(feature)

    print(f"\n   Feature breakdown:")
    print(f"   - Points: {len(points)}")
    print(f"   - LineStrings (tracks): {len(linestrings)}")
    print(f"   - Other: {len(other)}")

    return points, linestrings, other

def extract_tracks_from_linestrings(linestrings):
    """Extract coordinate tracks from LineString features"""
    print(f"\n🔍 Analyzing {len(linestrings)} LineString tracks...")

    tracks = []

    for idx, ls in enumerate(linestrings):
        coords = ls.get('geometry', {}).get('coordinates', [])
        props = ls.get('properties', {})

        if len(coords) > 0:
            # Convert to DataFrame
            if len(coords[0]) == 3:  # lon, lat, elevation
                df = pd.DataFrame(coords, columns=['longitude', 'latitude', 'altitude'])
            else:  # lon, lat
                df = pd.DataFrame(coords, columns=['longitude', 'latitude'])

            # Calculate basic stats
            track_info = {
                'index': idx,
                'points': len(coords),
                'properties': props,
                'data': df,
                'lon_range': (df['longitude'].min(), df['longitude'].max()),
                'lat_range': (df['latitude'].min(), df['latitude'].max()),
                'lon_mean': df['longitude'].mean(),
                'lat_mean': df['latitude'].mean()
            }

            tracks.append(track_info)

            print(f"\n   Track {idx}:")
            print(f"   - Points: {len(coords)}")
            print(f"   - Lon range: {track_info['lon_range'][0]:.6f} to {track_info['lon_range'][1]:.6f}")
            print(f"   - Lat range: {track_info['lat_range'][0]:.6f} to {track_info['lat_range'][1]:.6f}")
            print(f"   - Properties: {props}")

    return tracks

def identify_watch_tracks(tracks):
    """Identify which tracks belong to which watch"""
    print(f"\n🔬 Identifying separate watch tracks...")

    if len(tracks) < 2:
        print("   ⚠ Less than 2 tracks found - cannot compare watches")
        return None, None

    # First, remove duplicate tracks (same number of points and same center)
    unique_tracks = []
    seen_signatures = set()

    for track in tracks:
        # Create signature: (points, lon_mean rounded, lat_mean rounded)
        signature = (track['points'],
                    round(track['lon_mean'], 6),
                    round(track['lat_mean'], 6))

        if signature not in seen_signatures:
            seen_signatures.add(signature)
            unique_tracks.append(track)
            print(f"   ✓ Unique track found: {track['points']} points at ({track['lon_mean']:.6f}, {track['lat_mean']:.6f})")
        else:
            print(f"   ⚠ Duplicate track skipped: Track {track['index']}")

    if len(unique_tracks) < 2:
        print("   ⚠ Less than 2 unique tracks found - cannot compare watches")
        return None, None

    # Sort by number of points to get the two main tracks
    sorted_tracks = sorted(unique_tracks, key=lambda x: x['points'], reverse=True)

    watch1 = sorted_tracks[0]
    watch2 = sorted_tracks[1]

    print(f"\n   Watch 1 (Track {watch1['index']}):")
    print(f"   - Points: {watch1['points']}")
    print(f"   - Center: ({watch1['lon_mean']:.6f}, {watch1['lat_mean']:.6f})")

    # Check for properties that indicate watch type
    if 'properties' in watch1:
        props = watch1['properties']
        if 'name' in props:
            print(f"   - Name: {props['name']}")
        if 'type' in props:
            print(f"   - Type: {props['type']}")

    print(f"\n   Watch 2 (Track {watch2['index']}):")
    print(f"   - Points: {watch2['points']}")
    print(f"   - Center: ({watch2['lon_mean']:.6f}, {watch2['lat_mean']:.6f})")

    # Calculate distance between track centers
    dist_deg = np.sqrt((watch1['lon_mean'] - watch2['lon_mean'])**2 +
                      (watch1['lat_mean'] - watch2['lat_mean'])**2)
    dist_m = dist_deg * 111000  # Rough conversion to meters

    print(f"\n   Track separation: ~{dist_m:.1f} meters")

    return watch1, watch2

def calculate_track_statistics(track_info):
    """Calculate detailed statistics for a track"""
    df = track_info['data']

    # Calculate distances between consecutive points
    distances = []
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i-1][['latitude', 'longitude']]
        lat2, lon2 = df.iloc[i][['latitude', 'longitude']]

        # Haversine distance
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        dist = 6371000 * c  # Earth radius in meters

        distances.append(dist)

    distances = np.array(distances)
    total_distance = np.sum(distances)

    stats = {
        'total_distance_m': total_distance,
        'mean_step_m': np.mean(distances),
        'median_step_m': np.median(distances),
        'max_step_m': np.max(distances),
        'distances': distances
    }

    return stats

def visualize_tracks(watch1, watch2, stats1, stats2):
    """Create comprehensive visualization of both tracks"""
    print(f"\n📊 Creating visualizations...")

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Both tracks overlaid
    ax1 = fig.add_subplot(gs[0, :])

    df1 = watch1['data']
    df2 = watch2['data']

    ax1.plot(df1['longitude'], df1['latitude'], 'b-', linewidth=2, alpha=0.7, label='Watch 1 (Garmin?)')
    ax1.plot(df2['longitude'], df2['latitude'], 'r-', linewidth=2, alpha=0.7, label='Watch 2 (Coros?)')

    # Mark start/end points
    ax1.plot(df1['longitude'].iloc[0], df1['latitude'].iloc[0], 'go', markersize=15, label='Start (W1)', zorder=10)
    ax1.plot(df1['longitude'].iloc[-1], df1['latitude'].iloc[-1], 'gx', markersize=15, label='End (W1)', zorder=10)
    ax1.plot(df2['longitude'].iloc[0], df2['latitude'].iloc[0], 'mo', markersize=12, label='Start (W2)', zorder=10)
    ax1.plot(df2['longitude'].iloc[-1], df2['latitude'].iloc[-1], 'mx', markersize=12, label='End (W2)', zorder=10)

    ax1.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax1.set_title('GPS Tracks from Both Watches (400m Run)', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Panel 2: Watch 1 alone
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df1['longitude'], df1['latitude'], 'b-', linewidth=1.5, alpha=0.7)
    ax2.plot(df1['longitude'].iloc[0], df1['latitude'].iloc[0], 'go', markersize=10)
    ax2.plot(df1['longitude'].iloc[-1], df1['latitude'].iloc[-1], 'rx', markersize=10)
    ax2.set_xlabel('Longitude', fontsize=10)
    ax2.set_ylabel('Latitude', fontsize=10)
    ax2.set_title(f'Watch 1 Only ({watch1["points"]} points)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Panel 3: Watch 2 alone
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df2['longitude'], df2['latitude'], 'r-', linewidth=1.5, alpha=0.7)
    ax3.plot(df2['longitude'].iloc[0], df2['latitude'].iloc[0], 'go', markersize=10)
    ax3.plot(df2['longitude'].iloc[-1], df2['latitude'].iloc[-1], 'rx', markersize=10)
    ax3.set_xlabel('Longitude', fontsize=10)
    ax3.set_ylabel('Latitude', fontsize=10)
    ax3.set_title(f'Watch 2 Only ({watch2["points"]} points)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Panel 4: Point-by-point distance between tracks
    ax4 = fig.add_subplot(gs[1, 2])

    # Calculate point-by-point differences
    min_len = min(len(df1), len(df2))
    point_distances = []

    for i in range(min_len):
        lat1, lon1 = df1.iloc[i][['latitude', 'longitude']]
        lat2, lon2 = df2.iloc[i][['latitude', 'longitude']]

        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        dist = 6371000 * c

        point_distances.append(dist)

    ax4.plot(point_distances, color='purple', linewidth=1.5)
    ax4.axhline(np.mean(point_distances), color='red', linestyle='--',
                label=f'Mean: {np.mean(point_distances):.2f} m')
    ax4.set_xlabel('Point #', fontsize=10)
    ax4.set_ylabel('Distance (m)', fontsize=10)
    ax4.set_title('Watch-to-Watch Distance', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Step distances Watch 1
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(stats1['distances'], color='blue', linewidth=1, alpha=0.7)
    ax5.axhline(stats1['mean_step_m'], color='red', linestyle='--')
    ax5.set_xlabel('Step #', fontsize=10)
    ax5.set_ylabel('Distance (m)', fontsize=10)
    ax5.set_title(f'Watch 1 Step Sizes\nTotal: {stats1["total_distance_m"]:.1f} m', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Panel 6: Step distances Watch 2
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(stats2['distances'], color='red', linewidth=1, alpha=0.7)
    ax6.axhline(stats2['mean_step_m'], color='blue', linestyle='--')
    ax6.set_xlabel('Step #', fontsize=10)
    ax6.set_ylabel('Distance (m)', fontsize=10)
    ax6.set_title(f'Watch 2 Step Sizes\nTotal: {stats2["total_distance_m"]:.1f} m', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Panel 7: Summary statistics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    summary_text = f"""
TRACK COMPARISON SUMMARY

Watch 1:
  Points: {watch1['points']}
  Distance: {stats1['total_distance_m']:.1f} m
  Mean step: {stats1['mean_step_m']:.3f} m

Watch 2:
  Points: {watch2['points']}
  Distance: {stats2['total_distance_m']:.1f} m
  Mean step: {stats2['mean_step_m']:.3f} m

Difference:
  Distance: {abs(stats1['total_distance_m'] - stats2['total_distance_m']):.1f} m
  Mean separation: {np.mean(point_distances):.2f} m
  Max separation: {np.max(point_distances):.2f} m

Target: 400 m track
"""

    ax7.text(0.1, 0.5, summary_text, transform=ax7.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.suptitle('Dual-Watch GPS Analysis: Garmin vs Coros', fontsize=18, fontweight='bold')

    return fig, point_distances

def save_clean_tracks(watch1, watch2):
    """Save cleaned tracks as CSV for precision analysis"""
    print(f"\n💾 Saving cleaned tracks...")

    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'gps_precision')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Watch 1
    df1 = watch1['data'].copy()
    if 'altitude' not in df1.columns:
        df1['altitude'] = 0
    df1['timestamp'] = pd.date_range('2022-04-27 15:44:53', periods=len(df1), freq='1s')

    file1 = os.path.join(results_dir, f'garmin_cleaned_{timestamp}.csv')
    df1.to_csv(file1, index=False)
    print(f"   ✓ Watch 1: {file1}")

    # Save Watch 2
    df2 = watch2['data'].copy()
    if 'altitude' not in df2.columns:
        df2['altitude'] = 0
    df2['timestamp'] = pd.date_range('2022-04-27 15:44:53', periods=len(df2), freq='1s')

    file2 = os.path.join(results_dir, f'coros_cleaned_{timestamp}.csv')
    df2.to_csv(file2, index=False)
    print(f"   ✓ Watch 2: {file2}")

    return file1, file2

def main():
    """Main analysis"""
    # Load the messy GPS data
    file_path = os.path.join(os.path.dirname(__file__), 'gps_dataset.json')

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # Parse GeoJSON
    points, linestrings, other = load_and_analyze_geojson(file_path)

    if len(linestrings) == 0:
        print("\n❌ No LineString tracks found in the data")
        return

    # Extract tracks
    tracks = extract_tracks_from_linestrings(linestrings)

    # Identify watches
    watch1, watch2 = identify_watch_tracks(tracks)

    if watch1 is None or watch2 is None:
        print("\n❌ Could not identify two separate tracks")
        return

    # Calculate statistics
    print(f"\n📈 Calculating track statistics...")
    stats1 = calculate_track_statistics(watch1)
    stats2 = calculate_track_statistics(watch2)

    print(f"\n   Watch 1: {stats1['total_distance_m']:.1f} m total distance")
    print(f"   Watch 2: {stats2['total_distance_m']:.1f} m total distance")
    print(f"   Difference: {abs(stats1['total_distance_m'] - stats2['total_distance_m']):.1f} m")

    # Visualize
    fig, point_distances = visualize_tracks(watch1, watch2, stats1, stats2)

    # Save figure
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'gps_precision')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_file = os.path.join(results_dir, f'dual_watch_comparison_{timestamp}.png')
    plt.savefig(fig_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved: {fig_file}")

    # Save cleaned tracks
    file1, file2 = save_clean_tracks(watch1, watch2)

    plt.show()

    # Now ready for precision analysis
    print(f"\n" + "="*70)
    print(f"   ✓ DATA CLEANED AND READY FOR PRECISION ANALYSIS")
    print(f"="*70)
    print(f"\n   Next step:")
    print(f"   python gps_precision_analysis.py --watch1 {os.path.basename(file1)} --watch2 {os.path.basename(file2)}")
    print(f"\n   This will apply the 7-layer precision cascade to your actual data!")

if __name__ == "__main__":
    main()
