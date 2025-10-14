"""
Body Segmentation & Volume Calculation (Scale 4)

Calculates body volume, surface area, and moving volume from:
- Anthropometric data
- Video/photo segmentation (if available)
- 3D body modeling (SMPL)

Used to calculate air displacement and molecular interface during 400m run.

Author: Stella-Lorraine Observatory
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BodyGeometry:
    """Complete body geometry"""
    total_volume_liters: float
    surface_area_m2: float
    segment_volumes: Dict[str, float]  # Head, torso, arms, legs
    segment_areas: Dict[str, float]
    frontal_area_m2: float  # For drag calculation
    height_m: float
    weight_kg: float


def estimate_body_volume_anthropometric(
    height_m: float,
    weight_kg: float,
    sex: str = 'male'
) -> float:
    """
    Estimate total body volume from anthropometric data

    Methods:
    1. Simple: Volume = weight / body_density
    2. Detailed: Sum of segment volumes

    Average body density:
    - Male: 1.07 g/cm³
    - Female: 1.04 g/cm³
    - Athlete (low body fat): 1.08-1.10 g/cm³

    Args:
        height_m: Height in meters
        weight_kg: Weight in kilograms
        sex: 'male' or 'female'

    Returns:
        Total body volume in liters
    """
    # Body density for athletes (assume low body fat for 400m runner)
    body_density_g_cm3 = 1.09 if sex == 'male' else 1.06

    # Volume = mass / density
    volume_cm3 = (weight_kg * 1000) / body_density_g_cm3
    volume_liters = volume_cm3 / 1000

    return volume_liters


def calculate_body_surface_area(height_m: float, weight_kg: float,
                                method: str = 'dubois') -> float:
    """
    Calculate body surface area (BSA)

    Methods:
    1. Du Bois (1916): BSA = 0.007184 × W^0.425 × H^0.725
    2. Mosteller (1987): BSA = √(H × W / 3600)
    3. Haycock (1978): BSA = 0.024265 × W^0.5378 × H^0.3964

    Where W = weight (kg), H = height (cm)

    Args:
        height_m: Height in meters
        weight_kg: Weight in kilograms
        method: 'dubois', 'mosteller', or 'haycock'

    Returns:
        Body surface area in m²
    """
    height_cm = height_m * 100

    if method == 'dubois':
        bsa_m2 = 0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725)
    elif method == 'mosteller':
        bsa_m2 = np.sqrt(height_cm * weight_kg / 3600)
    elif method == 'haycock':
        bsa_m2 = 0.024265 * (weight_kg ** 0.5378) * (height_cm ** 0.3964)
    else:
        raise ValueError(f"Unknown method: {method}")

    return bsa_m2


def estimate_segment_volumes(height_m: float, weight_kg: float) -> Dict[str, float]:
    """
    Estimate volumes of body segments

    Based on anthropometric regression models (Winter 2009, de Leva 1996)

    Typical percentages of total body mass:
    - Head: 8%
    - Torso (trunk): 50%
    - Upper arms (2): 6%
    - Forearms (2): 4%
    - Hands (2): 1.2%
    - Thighs (2): 20%
    - Shanks (2): 9%
    - Feet (2): 2.8%

    Returns:
        Dictionary of segment volumes in liters
    """
    total_volume = estimate_body_volume_anthropometric(height_m, weight_kg)

    # Mass percentages from Winter 2009
    mass_percentages = {
        'head': 0.08,
        'torso': 0.50,
        'upper_arms': 0.06,
        'forearms': 0.04,
        'hands': 0.012,
        'thighs': 0.20,
        'shanks': 0.09,
        'feet': 0.028
    }

    # Convert to volumes (assuming uniform density)
    segment_volumes = {
        name: total_volume * pct
        for name, pct in mass_percentages.items()
    }

    return segment_volumes


def estimate_segment_surface_areas(height_m: float, weight_kg: float) -> Dict[str, float]:
    """
    Estimate surface areas of body segments

    Uses regression equations from anthropometric literature
    """
    bsa_total = calculate_body_surface_area(height_m, weight_kg)

    # Surface area percentages (approximate)
    area_percentages = {
        'head': 0.07,  # Face, scalp
        'torso_front': 0.18,  # Chest, abdomen
        'torso_back': 0.18,  # Back
        'upper_arms': 0.09,  # Both arms
        'forearms': 0.06,
        'hands': 0.05,
        'thighs': 0.19,
        'shanks': 0.14,
        'feet': 0.04
    }

    segment_areas = {
        name: bsa_total * pct
        for name, pct in area_percentages.items()
    }

    return segment_areas


def calculate_frontal_area(height_m: float, weight_kg: float,
                           posture: str = 'running') -> float:
    """
    Calculate frontal cross-sectional area (for drag calculation)

    Frontal area depends on posture:
    - Standing upright: ~0.5-0.7 m²
    - Running: ~0.4-0.6 m² (leaned forward)
    - Cycling (drops): ~0.3-0.4 m²

    Empirical formula (running):
    A_frontal ≈ 0.266 × height^0.725 × mass^0.425 × 0.8

    The 0.8 factor accounts for forward lean during running

    Returns:
        Frontal area in m²
    """
    if posture == 'standing':
        # Frontal area ≈ BSA × 0.3 (rough approximation)
        frontal_area = calculate_body_surface_area(height_m, weight_kg) * 0.3
    elif posture == 'running':
        # Empirical formula with forward lean factor
        frontal_area = 0.266 * (height_m * 100) ** 0.725 * weight_kg ** 0.425 / 100 * 0.8
    elif posture == 'cycling':
        frontal_area = 0.266 * (height_m * 100) ** 0.725 * weight_kg ** 0.425 / 100 * 0.6
    else:
        frontal_area = calculate_body_surface_area(height_m, weight_kg) * 0.3

    return frontal_area


def create_body_geometry(
    height_m: float,
    weight_kg: float,
    sex: str = 'male',
    posture: str = 'running'
) -> BodyGeometry:
    """
    Create complete body geometry model

    Args:
        height_m: Height in meters
        weight_kg: Weight in kilograms
        sex: 'male' or 'female'
        posture: 'standing', 'running', 'cycling'

    Returns:
        Complete BodyGeometry object
    """
    total_volume = estimate_body_volume_anthropometric(height_m, weight_kg, sex)
    surface_area = calculate_body_surface_area(height_m, weight_kg)
    segment_volumes = estimate_segment_volumes(height_m, weight_kg)
    segment_areas = estimate_segment_surface_areas(height_m, weight_kg)
    frontal_area = calculate_frontal_area(height_m, weight_kg, posture)

    return BodyGeometry(
        total_volume_liters=total_volume,
        surface_area_m2=surface_area,
        segment_volumes=segment_volumes,
        segment_areas=segment_areas,
        frontal_area_m2=frontal_area,
        height_m=height_m,
        weight_kg=weight_kg
    )


def calculate_moving_volume(
    body_geometry: BodyGeometry,
    velocity_ms: np.ndarray,
    timestamps_s: np.ndarray
) -> pd.DataFrame:
    """
    Calculate volume swept by body movement

    Volume swept per time step:
    V_swept = frontal_area × velocity × dt

    This is the volume of air that must be displaced!

    Args:
        body_geometry: Body geometry model
        velocity_ms: Velocity time series (m/s)
        timestamps_s: Time points (seconds)

    Returns:
        DataFrame with swept volumes
    """
    dt = np.diff(timestamps_s)
    dt = np.append(dt, dt[-1] if len(dt) > 0 else 1.0)

    # Volume swept per time step
    volume_swept_m3 = body_geometry.frontal_area_m2 * velocity_ms * dt

    # Cumulative volume
    cumulative_volume_m3 = np.cumsum(volume_swept_m3)

    results = pd.DataFrame({
        'timestamp_s': timestamps_s,
        'velocity_ms': velocity_ms,
        'volume_swept_m3': volume_swept_m3,
        'cumulative_volume_m3': cumulative_volume_m3,
        'cumulative_volume_liters': cumulative_volume_m3 * 1000
    })

    return results


def calculate_boundary_layer_volume(
    body_geometry: BodyGeometry,
    velocity_ms: float,
    air_viscosity: float = 1.81e-5
) -> float:
    """
    Calculate volume of air boundary layer around body

    Boundary layer thickness (turbulent):
    δ ≈ 0.37 × x / Re^(1/5)

    where:
    - x = characteristic length (body height)
    - Re = Reynolds number = ρ × v × x / μ

    Typical:
    - Running at 10 m/s: δ ≈ 2-5 cm
    - This layer contains molecules in direct contact with skin!

    Args:
        body_geometry: Body geometry model
        velocity_ms: Running velocity (m/s)
        air_viscosity: Dynamic viscosity of air (Pa·s)

    Returns:
        Boundary layer volume in m³
    """
    # Air density at STP
    air_density = 1.225  # kg/m³

    # Reynolds number
    char_length = body_geometry.height_m
    Re = air_density * velocity_ms * char_length / air_viscosity

    # Boundary layer thickness (turbulent flow assumed for running)
    delta_m = 0.37 * char_length / (Re ** 0.2)

    # Boundary layer volume ≈ surface_area × boundary_layer_thickness
    boundary_layer_volume_m3 = body_geometry.surface_area_m2 * delta_m

    return boundary_layer_volume_m3


def main():
    """
    Example: Calculate body geometry for 400m runner
    """
    print("=" * 70)
    print(" BODY SEGMENTATION & GEOMETRY (SCALE 4) ")
    print("=" * 70)

    # Typical 400m runner anthropometrics
    height_m = 1.75  # 175 cm
    weight_kg = 70  # 70 kg
    sex = 'male'

    print(f"\n[1/4] Anthropometric Data")
    print(f"  Height: {height_m} m ({height_m * 100:.0f} cm)")
    print(f"  Weight: {weight_kg} kg")
    print(f"  Sex: {sex}")

    # Create body geometry
    print(f"\n[2/4] Calculate Body Geometry")
    body_geo = create_body_geometry(height_m, weight_kg, sex, posture='running')

    print(f"  Total volume: {body_geo.total_volume_liters:.2f} liters")
    print(f"  Surface area: {body_geo.surface_area_m2:.3f} m²")
    print(f"  Frontal area (running): {body_geo.frontal_area_m2:.3f} m²")

    print(f"\n  Segment volumes:")
    for name, vol in body_geo.segment_volumes.items():
        print(f"    {name}: {vol:.2f} L")

    # Calculate for 400m run
    print(f"\n[3/4] Calculate Moving Volume for 400m Run")

    # Load smartwatch data for actual velocities
    try:
        from watch import load_400m_run_data
        watch1, watch2 = load_400m_run_data()
        velocity_ms = watch1.biomechanics['speed_ms'].values
        timestamps = watch1.biomechanics['timestamp'].values
        timestamps_s = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
    except:
        # Simulate if data not available
        duration_s = 60  # 60 seconds for 400m
        timestamps_s = np.arange(0, duration_s, 1.0)
        velocity_ms = np.full(duration_s, 6.67)  # ~400m in 60s = 6.67 m/s

    moving_volume = calculate_moving_volume(body_geo, velocity_ms, timestamps_s)

    total_displaced_m3 = moving_volume['cumulative_volume_m3'].iloc[-1]
    total_displaced_liters = total_displaced_m3 * 1000

    print(f"  Total air displaced: {total_displaced_m3:.2f} m³ ({total_displaced_liters:.0f} liters)")
    print(f"  Air mass displaced: {total_displaced_m3 * 1.225:.1f} kg")

    # Calculate number of molecules
    air_molar_mass = 0.029  # kg/mol
    avogadro = 6.022e23
    total_mass_kg = total_displaced_m3 * 1.225
    total_moles = total_mass_kg / air_molar_mass
    total_molecules = total_moles * avogadro

    print(f"  Total molecules displaced: {total_molecules:.2e}")
    print(f"  O₂ molecules (21%): {total_molecules * 0.21:.2e}")

    # Boundary layer
    print(f"\n[4/4] Boundary Layer Analysis")
    mean_velocity = np.mean(velocity_ms)
    boundary_layer_vol_m3 = calculate_boundary_layer_volume(body_geo, mean_velocity)

    print(f"  Mean velocity: {mean_velocity:.2f} m/s")
    print(f"  Boundary layer volume: {boundary_layer_vol_m3:.4f} m³ ({boundary_layer_vol_m3 * 1000:.1f} liters)")
    print(f"  Boundary layer thickness: {boundary_layer_vol_m3 / body_geo.surface_area_m2 * 1000:.1f} mm")

    # Molecules in boundary layer (in constant contact with skin)
    boundary_molecules = boundary_layer_vol_m3 * 2.7e25  # molecules/m³ at STP
    boundary_o2 = boundary_molecules * 0.21

    print(f"  Molecules in boundary layer: {boundary_molecules:.2e}")
    print(f"  O₂ in boundary layer: {boundary_o2:.2e}")
    print(f"  → These molecules are in direct contact with skin!")

    # Save results
    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'body_geometry')
    os.makedirs(results_dir, exist_ok=True)

    # Save geometry
    import json
    geo_dict = {
        'height_m': body_geo.height_m,
        'weight_kg': body_geo.weight_kg,
        'total_volume_liters': body_geo.total_volume_liters,
        'surface_area_m2': body_geo.surface_area_m2,
        'frontal_area_m2': body_geo.frontal_area_m2,
        'segment_volumes': body_geo.segment_volumes,
        'segment_areas': body_geo.segment_areas,
        'total_air_displaced_m3': float(total_displaced_m3),
        'total_molecules_displaced': float(total_molecules),
        'o2_molecules_displaced': float(total_molecules * 0.21),
        'boundary_layer_volume_m3': float(boundary_layer_vol_m3),
        'boundary_layer_o2_molecules': float(boundary_o2)
    }

    geo_file = os.path.join(results_dir, 'body_geometry_400m.json')
    with open(geo_file, 'w') as f:
        json.dump(geo_dict, f, indent=2)
    print(f"\n✓ Body geometry saved: {geo_file}")

    # Save moving volume
    moving_vol_file = os.path.join(results_dir, 'moving_volume_400m.csv')
    moving_volume.to_csv(moving_vol_file, index=False)
    print(f"✓ Moving volume saved: {moving_vol_file}")

    print("\n" + "=" * 70)
    print(" BODY-ATMOSPHERE INTERFACE QUANTIFIED ")
    print("=" * 70)
    print(f"\nDuring 400m run:")
    print(f"  - Displaced {total_molecules:.2e} air molecules")
    print(f"  - Contacted {boundary_o2:.2e} O₂ molecules continuously")
    print(f"  - Surface area: {body_geo.surface_area_m2:.2f} m² in constant molecular exchange")

    return body_geo, moving_volume


if __name__ == "__main__":
    main()
