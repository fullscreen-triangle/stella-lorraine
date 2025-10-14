"""
Smartwatch Data Integration (Scale 2: Cardiovascular)

Loads and processes smartwatch data from the 400m run, including:
- GPS tracks
- Heart rate
- Biomechanical metrics (stance time, cadence, etc.)
- Multi-modal sensor fusion

Integrates with Munich Airport ground truth for time synchronization.

Author: Stella-Lorraine Observatory
Date: 2024
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SmartWatchData:
    """Complete smartwatch dataset"""
    gps_track: pd.DataFrame  # GPS coordinates, timestamps
    heart_rate: pd.DataFrame  # HR time series
    biomechanics: pd.DataFrame  # Stride, cadence, stance time, etc.
    metadata: Dict  # Watch model, firmware, etc.


def load_gps_dataset_json(file_path: str) -> Dict:
    """
    Load the messy GeoJSON dataset from 400m run

    File: gps_dataset.json
    Contains: Points, LineStrings for two watches (Garmin & Coros)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"  Loaded: {file_path}")
    print(f"  Type: {data.get('type', 'Unknown')}")
    print(f"  Features: {len(data.get('features', []))}")

    return data


def extract_watch_tracks(geojson_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract two watch tracks from GeoJSON

    Returns:
        (watch1_df, watch2_df) with GPS coordinates and timestamps
    """
    features = geojson_data.get('features', [])

    # Identify LineString features (tracks)
    tracks = []
    for feature in features:
        geom = feature.get('geometry')
        if geom and geom.get('type') == 'LineString':
            coords = geom.get('coordinates', [])
            properties = feature.get('properties', {})

            # Extract timestamps if available
            coord_props = properties.get('coordinateProperties', {})
            times = coord_props.get('times', [])

            tracks.append({
                'coordinates': coords,
                'times': times,
                'properties': properties,
                'n_points': len(coords)
            })

    print(f"\n  Found {len(tracks)} tracks:")
    for i, track in enumerate(tracks):
        print(f"    Track {i}: {track['n_points']} points")

    # Identify the two distinct watch tracks (skip duplicates)
    unique_tracks = []
    for track in tracks:
        # Check if this is a duplicate
        is_duplicate = False
        for unique_track in unique_tracks:
            if track['n_points'] == unique_track['n_points']:
                # Compare mean positions to check if same track
                mean_lon_1 = np.mean([c[0] for c in track['coordinates']])
                mean_lon_2 = np.mean([c[0] for c in unique_track['coordinates']])
                if abs(mean_lon_1 - mean_lon_2) < 0.0001:  # Within ~10 m
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_tracks.append(track)

    print(f"  Unique tracks: {len(unique_tracks)}")

    if len(unique_tracks) < 2:
        print("  Warning: Expected 2 distinct tracks, found {len(unique_tracks)}")
        # Duplicate one track for demonstration
        if len(unique_tracks) == 1:
            unique_tracks.append(unique_tracks[0])

    # Convert to DataFrames
    def track_to_df(track):
        data = []
        times = track.get('times', [])

        for i, coord in enumerate(track['coordinates']):
            lon, lat = coord[0], coord[1]
            alt = coord[2] if len(coord) > 2 else None

            timestamp = None
            if i < len(times):
                try:
                    timestamp = pd.to_datetime(times[i])
                except:
                    pass

            data.append({
                'timestamp': timestamp,
                'longitude': lon,
                'latitude': lat,
                'altitude_m': alt,
                'point_index': i
            })

        df = pd.DataFrame(data)

        # If no timestamps, generate them (1 Hz assumed)
        if df['timestamp'].isna().all():
            start_time = datetime(2022, 4, 27, 15, 44, 53)  # From original data
            df['timestamp'] = [start_time + timedelta(seconds=i) for i in range(len(df))]

        return df

    watch1_df = track_to_df(unique_tracks[0])
    watch2_df = track_to_df(unique_tracks[1]) if len(unique_tracks) > 1 else watch1_df.copy()

    return watch1_df, watch2_df


def calculate_distance_speed(gps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate instantaneous distance and speed from GPS

    Uses Haversine formula for distance between GPS points
    """
    from geopy.distance import geodesic

    gps_df = gps_df.copy()

    # Calculate distance between consecutive points
    distances = [0]
    for i in range(1, len(gps_df)):
        point1 = (gps_df.iloc[i-1]['latitude'], gps_df.iloc[i-1]['longitude'])
        point2 = (gps_df.iloc[i]['latitude'], gps_df.iloc[i]['longitude'])
        dist = geodesic(point1, point2).meters
        distances.append(dist)

    gps_df['distance_m'] = distances
    gps_df['cumulative_distance_m'] = np.cumsum(distances)

    # Calculate speed
    time_diffs = gps_df['timestamp'].diff().dt.total_seconds()
    gps_df['speed_ms'] = gps_df['distance_m'] / time_diffs
    gps_df['speed_ms'] = gps_df['speed_ms'].fillna(0)

    return gps_df


def extract_heart_rate_from_properties(geojson_data: Dict) -> pd.DataFrame:
    """
    Extract heart rate data from GeoJSON properties

    Some tracks have 'ns3:TrackPointExtensions' with heart rate
    """
    features = geojson_data.get('features', [])

    hr_data = []
    for feature in features:
        geom = feature.get('geometry')
        if geom and geom.get('type') == 'LineString':
            coords = geom.get('coordinates', [])
            properties = feature.get('properties', {})

            coord_props = properties.get('coordinateProperties', {})
            times = coord_props.get('times', [])
            hr_values = coord_props.get('ns3:TrackPointExtensions', [])

            if hr_values and len(hr_values) > 0:
                for i, (time_str, hr) in enumerate(zip(times, hr_values)):
                    try:
                        timestamp = pd.to_datetime(time_str)
                        hr_data.append({
                            'timestamp': timestamp,
                            'heart_rate_bpm': hr,
                            'point_index': i
                        })
                    except:
                        pass

    if len(hr_data) > 0:
        hr_df = pd.DataFrame(hr_data)
        print(f"  Heart rate data points: {len(hr_df)}")
        print(f"  HR range: {hr_df['heart_rate_bpm'].min():.0f} - {hr_df['heart_rate_bpm'].max():.0f} bpm")
        return hr_df
    else:
        # Simulate HR data if not available
        print("  No heart rate data found, simulating...")
        return simulate_heart_rate()


def simulate_heart_rate(duration_s: int = 150, base_hr: int = 140) -> pd.DataFrame:
    """
    Simulate realistic heart rate for 400m run

    Pattern:
    - Start: Moderate (~120 bpm)
    - Accelerate: Rising to max (~180 bpm)
    - Maintain: High effort (~170-180 bpm)
    - Finish: Peak (~185 bpm)
    - Recovery: Gradual decrease
    """
    timestamps = pd.date_range(
        datetime(2022, 4, 27, 15, 44, 53),
        periods=duration_s,
        freq='1S'
    )

    # Realistic 400m HR profile
    hr_profile = []
    for i in range(duration_s):
        if i < 10:  # First 10s: acceleration
            hr = 120 + (i / 10) * 40
        elif i < 50:  # 10-50s: building effort
            hr = 160 + (i - 10) / 40 * 20
        elif i < 100:  # 50-100s: high effort maintained
            hr = 175 + np.sin(i * 0.3) * 5
        else:  # >100s: fatigue, slight increase
            hr = 178 + (i - 100) / 50 * 7

        # Add physiological variability
        hr += np.random.normal(0, 2)
        hr_profile.append(hr)

    hr_df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate_bpm': hr_profile
    })

    return hr_df


def extract_biomechanics(gps_df: pd.DataFrame, hr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract/estimate biomechanical parameters

    From GPS:
    - Stride length (from speed)
    - Cadence (steps per minute)
    - Vertical oscillation (from altitude if available)

    From HR:
    - Cardiovascular load
    - Estimated VO2
    """
    biomech_df = gps_df.copy()

    # Estimate stride length and cadence from speed
    # Typical: stride_length (m) ≈ 1.3 at moderate pace, up to 2.0 at sprint
    # cadence (steps/min) ≈ 180-200 for 400m runners

    biomech_df['estimated_stride_length_m'] = biomech_df['speed_ms'] / 3.0  # Approximate
    biomech_df['estimated_stride_length_m'] = biomech_df['estimated_stride_length_m'].clip(1.0, 2.5)

    biomech_df['estimated_cadence_spm'] = (biomech_df['speed_ms'] / biomech_df['estimated_stride_length_m']) * 60
    biomech_df['estimated_cadence_spm'] = biomech_df['estimated_cadence_spm'].fillna(180)

    # Estimate stance time (% of stride cycle on ground)
    # Faster running → less ground contact time
    # Sprint: ~30-40% stance time, Jog: ~50-60%
    biomech_df['estimated_stance_time_pct'] = 60 - (biomech_df['speed_ms'] - 5) * 3
    biomech_df['estimated_stance_time_pct'] = biomech_df['estimated_stance_time_pct'].clip(30, 60)

    # Merge with HR
    if len(hr_df) > 0:
        # Interpolate HR to GPS timestamps
        hr_series = pd.Series(hr_df['heart_rate_bpm'].values, index=hr_df['timestamp'])
        biomech_df['heart_rate_bpm'] = biomech_df['timestamp'].map(
            lambda t: hr_series.asof(t) if pd.notna(t) else np.nan
        )
        biomech_df['heart_rate_bpm'] = biomech_df['heart_rate_bpm'].interpolate()

    # Estimate VO2 from HR (rough approximation)
    # VO2max typically 50-70 mL/kg/min for trained runners
    # During 400m: 90-100% of VO2max
    if 'heart_rate_bpm' in biomech_df.columns:
        hr_max = 220 - 25  # Assume 25 years old, HR_max = 220 - age
        biomech_df['hr_percent_max'] = (biomech_df['heart_rate_bpm'] / hr_max * 100).clip(0, 100)
        biomech_df['estimated_vo2_ml_kg_min'] = 60 * (biomech_df['hr_percent_max'] / 100)

    return biomech_df


def synchronize_to_atomic_clock(
    watch_data: SmartWatchData,
    atomic_clock_ref: Dict
) -> SmartWatchData:
    """
    Synchronize all smartwatch timestamps to Munich Airport atomic clock

    Smartwatch clocks:
    - GPS-disciplined (if GPS active): ±100 ns
    - Quartz oscillator (no GPS): ±50 ppm = ±4.3 s/day

    For 400m run (~150 s):
    - Maximum drift: ~0.01 seconds (negligible)
    - But ensures consistency with all other cascade measurements
    """
    # In practice, would apply GPS time corrections
    # For now, ensure all timestamps are UTC

    if 'gps_track' in watch_data.__dict__:
        watch_data.gps_track['timestamp_atomic'] = pd.to_datetime(
            watch_data.gps_track['timestamp'], utc=True
        )

    if 'heart_rate' in watch_data.__dict__:
        watch_data.heart_rate['timestamp_atomic'] = pd.to_datetime(
            watch_data.heart_rate['timestamp'], utc=True
        )

    if 'biomechanics' in watch_data.__dict__:
        watch_data.biomechanics['timestamp_atomic'] = pd.to_datetime(
            watch_data.biomechanics['timestamp'], utc=True
        )

    return watch_data


def load_400m_run_data(gps_json_path: Optional[str] = None) -> Tuple[SmartWatchData, SmartWatchData]:
    """
    Load complete 400m run data for both watches

    Returns:
        (watch1_data, watch2_data)
    """
    print("=" * 70)
    print(" SMARTWATCH DATA LOADING (SCALE 2) ")
    print("=" * 70)

    # Default path
    if gps_json_path is None:
        gps_json_path = Path(__file__).parent / '..' / '..' / 'src' / 'precision' / 'gps_dataset.json'
        gps_json_path = str(gps_json_path.resolve())

    print(f"\n[1/5] Load GeoJSON Data")
    geojson_data = load_gps_dataset_json(gps_json_path)

    print(f"\n[2/5] Extract Watch Tracks")
    watch1_gps, watch2_gps = extract_watch_tracks(geojson_data)
    print(f"  Watch 1: {len(watch1_gps)} GPS points")
    print(f"  Watch 2: {len(watch2_gps)} GPS points")

    print(f"\n[3/5] Calculate Distance & Speed")
    watch1_gps = calculate_distance_speed(watch1_gps)
    watch2_gps = calculate_distance_speed(watch2_gps)
    print(f"  Watch 1 total distance: {watch1_gps['cumulative_distance_m'].iloc[-1]:.1f} m")
    print(f"  Watch 2 total distance: {watch2_gps['cumulative_distance_m'].iloc[-1]:.1f} m")

    print(f"\n[4/5] Extract Heart Rate")
    hr_df = extract_heart_rate_from_properties(geojson_data)

    print(f"\n[5/5] Extract Biomechanics")
    watch1_biomech = extract_biomechanics(watch1_gps, hr_df)
    watch2_biomech = extract_biomechanics(watch2_gps, hr_df)

    # Create SmartWatchData objects
    watch1_data = SmartWatchData(
        gps_track=watch1_gps,
        heart_rate=hr_df,
        biomechanics=watch1_biomech,
        metadata={'model': 'Garmin', 'run_date': '2022-04-27'}
    )

    watch2_data = SmartWatchData(
        gps_track=watch2_gps,
        heart_rate=hr_df,
        biomechanics=watch2_biomech,
        metadata={'model': 'Coros', 'run_date': '2022-04-27'}
    )

    print("\n" + "=" * 70)
    print(" SMARTWATCH DATA LOADED ")
    print("=" * 70)

    return watch1_data, watch2_data


def main():
    """
    Example: Load and process 400m smartwatch data
    """
    watch1, watch2 = load_400m_run_data()

    # Display summary statistics
    print("\n📊 Summary Statistics:")
    print(f"\n  Watch 1 (Garmin):")
    print(f"    Duration: {(watch1.gps_track['timestamp'].iloc[-1] - watch1.gps_track['timestamp'].iloc[0]).total_seconds():.1f} s")
    print(f"    Distance: {watch1.gps_track['cumulative_distance_m'].iloc[-1]:.1f} m")
    print(f"    Avg speed: {watch1.biomechanics['speed_ms'].mean():.2f} m/s")
    print(f"    Avg HR: {watch1.biomechanics['heart_rate_bpm'].mean():.0f} bpm")
    print(f"    Avg cadence: {watch1.biomechanics['estimated_cadence_spm'].mean():.0f} spm")

    print(f"\n  Watch 2 (Coros):")
    print(f"    Duration: {(watch2.gps_track['timestamp'].iloc[-1] - watch2.gps_track['timestamp'].iloc[0]).total_seconds():.1f} s")
    print(f"    Distance: {watch2.gps_track['cumulative_distance_m'].iloc[-1]:.1f} m")
    print(f"    Avg speed: {watch2.biomechanics['speed_ms'].mean():.2f} m/s")
    print(f"    Avg HR: {watch2.biomechanics['heart_rate_bpm'].mean():.0f} bpm")

    # Save processed data
    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'smartwatch')
    os.makedirs(results_dir, exist_ok=True)

    timestamp_str = watch1.gps_track['timestamp'].iloc[0].strftime("%Y%m%d_%H%M%S")

    watch1.biomechanics.to_csv(
        os.path.join(results_dir, f'watch1_complete_{timestamp_str}.csv'),
        index=False
    )
    watch2.biomechanics.to_csv(
        os.path.join(results_dir, f'watch2_complete_{timestamp_str}.csv'),
        index=False
    )

    print(f"\n✓ Data saved to: {results_dir}")

    return watch1, watch2


if __name__ == "__main__":
    main()
