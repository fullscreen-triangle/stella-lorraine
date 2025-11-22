# interferometry/angular_resolution.py
# git pat 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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.constants as const
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ResolutionMetrics:
    """Angular resolution metrics"""
    wavelength: float  # [m]
    baseline_length: float  # [m]
    angular_resolution_rad: float  # [rad]
    angular_resolution_arcsec: float  # [arcsec]
    angular_resolution_microarcsec: float  # [μas]
    diffraction_limit: float  # [rad]
    improvement_over_single_aperture: float  # dimensionless


class AngularResolutionCalculator:
    """
    Calculate angular resolution for interferometric systems

    Key relation: θ_min ≈ λ/D

    For trans-Planckian baselines (D = 10⁴ km), achieves
    θ ~ 10⁻⁵ microarcseconds at λ = 500 nm
    """

    def __init__(self, wavelength: float):
        """
        Args:
            wavelength: Observation wavelength [m]
        """
        self.lambda_ = wavelength
        self.k = 2 * np.pi / wavelength

    def rayleigh_criterion(self, baseline: float) -> float:
        """
        Rayleigh criterion for angular resolution

        θ = 1.22 λ/D (for circular aperture)
        θ = λ/D (for interferometer baseline)

        Args:
            baseline: Baseline length [m]

        Returns:
            Angular resolution [rad]
        """
        return self.lambda_ / baseline

    def angular_resolution_arcsec(self, baseline: float) -> float:
        """
        Angular resolution in arcseconds

        Args:
            baseline: Baseline length [m]

        Returns:
            Angular resolution [arcsec]
        """
        theta_rad = self.rayleigh_criterion(baseline)
        return theta_rad * (180 * 3600 / np.pi)

    def angular_resolution_microarcsec(self, baseline: float) -> float:
        """
        Angular resolution in microarcseconds

        Args:
            baseline: Baseline length [m]

        Returns:
            Angular resolution [μas]
        """
        return self.angular_resolution_arcsec(baseline) * 1e6

    def calculate_metrics(self,
                         baseline: float,
                         aperture_diameter: float = 1.0) -> ResolutionMetrics:
        """
        Calculate complete resolution metrics

        Args:
            baseline: Interferometer baseline [m]
            aperture_diameter: Single aperture diameter [m]

        Returns:
            ResolutionMetrics
        """
        # Interferometric resolution
        theta_rad = self.rayleigh_criterion(baseline)
        theta_arcsec = self.angular_resolution_arcsec(baseline)
        theta_microarcsec = self.angular_resolution_microarcsec(baseline)

        # Single aperture diffraction limit
        theta_diffraction = self.rayleigh_criterion(aperture_diameter)

        # Improvement factor
        improvement = theta_diffraction / theta_rad

        return ResolutionMetrics(
            wavelength=self.lambda_,
            baseline_length=baseline,
            angular_resolution_rad=theta_rad,
            angular_resolution_arcsec=theta_arcsec,
            angular_resolution_microarcsec=theta_microarcsec,
            diffraction_limit=theta_diffraction,
            improvement_over_single_aperture=improvement
        )

    def spatial_resolution_at_distance(self,
                                      baseline: float,
                                      distance: float) -> float:
        """
        Calculate spatial resolution at given distance

        Δx = θ × d

        Args:
            baseline: Baseline length [m]
            distance: Distance to target [m]

        Returns:
            Spatial resolution [m]
        """
        theta = self.rayleigh_criterion(baseline)
        return theta * distance

    def exoplanet_imaging_capability(self,
                                    baseline: float,
                                    planet_distance_pc: float,
                                    planet_radius_earth: float = 1.0) -> dict:
        """
        Assess capability to image exoplanet

        Args:
            baseline: Baseline length [m]
            planet_distance_pc: Distance to planet [parsec]
            planet_radius_earth: Planet radius [Earth radii]

        Returns:
            Dictionary with imaging metrics
        """
        # Convert to SI units
        pc_to_m = 3.086e16
        R_earth = 6.371e6  # m

        distance = planet_distance_pc * pc_to_m
        planet_radius = planet_radius_earth * R_earth

        # Angular size of planet
        theta_planet = planet_radius / distance

        # Resolution
        theta_resolution = self.rayleigh_criterion(baseline)

        # Spatial resolution at planet
        spatial_res = self.spatial_resolution_at_distance(baseline, distance)

        # Number of resolution elements across planet
        resolution_elements = planet_radius / spatial_res

        # Can we resolve it?
        resolvable = theta_planet > theta_resolution

        # Can we image surface features?
        # Need ~10 resolution elements for meaningful imaging
        imageable = resolution_elements > 10

        return {
            'planet_distance_pc': planet_distance_pc,
            'planet_distance_m': distance,
            'planet_radius_m': planet_radius,
            'planet_angular_size_rad': theta_planet,
            'planet_angular_size_microarcsec': theta_planet * (180*3600/np.pi) * 1e6,
            'resolution_rad': theta_resolution,
            'resolution_microarcsec': theta_resolution * (180*3600/np.pi) * 1e6,
            'spatial_resolution_m': spatial_res,
            'spatial_resolution_km': spatial_res / 1e3,
            'resolution_elements_across_planet': resolution_elements,
            'resolvable': resolvable,
            'imageable': imageable
        }

    def uv_coverage_resolution(self,
                              baselines: List[np.ndarray],
                              source_declination: float = 0.0) -> dict:
        """
        Calculate resolution from UV coverage

        Args:
            baselines: List of baseline vectors [m]
            source_declination: Source declination [rad]

        Returns:
            Dictionary with UV coverage metrics
        """
        # Convert baselines to UV coordinates
        u_coords = []
        v_coords = []

        for baseline in baselines:
            # Project baseline onto UV plane
            # u = baseline · east / λ
            # v = baseline · north / λ
            u = baseline[0] / self.lambda_
            v = baseline[1] / self.lambda_
            u_coords.append(u)
            v_coords.append(v)

        u_coords = np.array(u_coords)
        v_coords = np.array(v_coords)

        # Maximum baseline in UV plane
        uv_max = np.max(np.sqrt(u_coords**2 + v_coords**2))

        # Angular resolution from UV coverage
        theta_u = 1.0 / (2 * uv_max)  # radians

        # UV coverage area (proxy for image quality)
        # Use convex hull area
        from scipy.spatial import ConvexHull
        try:
            points = np.column_stack([u_coords, v_coords])
            hull = ConvexHull(points)
            uv_area = hull.volume  # In 2D, volume = area
        except:
            uv_area = 0.0

        return {
            'num_baselines': len(baselines),
            'u_coords': u_coords,
            'v_coords': v_coords,
            'max_uv_distance': uv_max,
            'angular_resolution_rad': theta_u,
            'angular_resolution_microarcsec': theta_u * (180*3600/np.pi) * 1e6,
            'uv_coverage_area': uv_area
        }


class TransPlanckianResolutionValidator:
    """
    Validate trans-Planckian baseline resolution claims

    Paper claims: θ ~ 10⁻⁵ μas with D = 10⁴ km at λ = 500 nm
    """

    def __init__(self):
        pass

    def validate_paper_claim(self,
                            wavelength: float = 500e-9,
                            baseline: float = 1e7) -> dict:
        """
        Validate paper's resolution claim

        Args:
            wavelength: Wavelength [m] (default 500 nm)
            baseline: Baseline [m] (default 10⁴ km)

        Returns:
            Validation dictionary
        """
        calc = AngularResolutionCalculator(wavelength)
        metrics = calc.calculate_metrics(baseline)

        # Paper claim
        paper_claim_microarcsec = 1e-5  # 10⁻⁵ μas

        # Calculated value
        calculated_microarcsec = metrics.angular_resolution_microarcsec

        # Comparison
        ratio = calculated_microarcsec / paper_claim_microarcsec
        agreement = 0.5 < ratio < 2.0  # Within factor of 2

        return {
            'wavelength_nm': wavelength * 1e9,
            'baseline_km': baseline / 1e3,
            'paper_claim_microarcsec': paper_claim_microarcsec,
            'calculated_microarcsec': calculated_microarcsec,
            'ratio': ratio,
            'agreement': agreement,
            'metrics': metrics
        }

    def compare_with_existing_instruments(self,
                                         wavelength: float = 500e-9,
                                         baseline: float = 1e7) -> dict:
        """
        Compare with existing interferometers

        Args:
            wavelength: Wavelength [m]
            baseline: Baseline [m]

        Returns:
            Comparison dictionary
        """
        calc = AngularResolutionCalculator(wavelength)

        # Trans-Planckian categorical interferometer
        trans_planckian = calc.calculate_metrics(baseline)

        # Existing instruments
        instruments = {
            'HST': calc.calculate_metrics(2.4),  # Hubble
            'VLT': calc.calculate_metrics(8.0),  # Very Large Telescope
            'VLTI': calc.calculate_metrics(200),  # VLT Interferometer
            'EHT': calc.calculate_metrics(1e7),  # Event Horizon Telescope (Earth diameter)
            'JWST': calc.calculate_metrics(6.5),  # James Webb
        }

        # Improvement factors
        improvements = {}
        for name, metrics in instruments.items():
            improvements[name] = metrics.angular_resolution_rad / trans_planckian.angular_resolution_rad

        return {
            'trans_planckian_resolution_microarcsec': trans_planckian.angular_resolution_microarcsec,
            'instrument_resolutions': {
                name: metrics.angular_resolution_microarcsec
                for name, metrics in instruments.items()
            },
            'improvement_factors': improvements
        }

    def exoplanet_survey_capability(self,
                                   baseline: float = 1e7,
                                   wavelength: float = 500e-9) -> dict:
        """
        Assess exoplanet imaging survey capability

        Args:
            baseline: Baseline [m]
            wavelength: Wavelength [m]

        Returns:
            Survey capability dictionary
        """
        calc = AngularResolutionCalculator(wavelength)

        # Test various scenarios
        scenarios = {
            'Earth_at_10pc': (10, 1.0),  # Earth-like at 10 parsec
            'Earth_at_100pc': (100, 1.0),
            'Jupiter_at_10pc': (10, 11.2),  # Jupiter-like
            'Super_Earth_at_5pc': (5, 2.0),
            'Hot_Jupiter_at_50pc': (50, 15.0)
        }

        results = {}
        for name, (distance_pc, radius_earth) in scenarios.items():
            capability = calc.exoplanet_imaging_capability(
                baseline, distance_pc, radius_earth
            )
            results[name] = capability

        # Summary statistics
        num_resolvable = sum(1 for r in results.values() if r['resolvable'])
        num_imageable = sum(1 for r in results.values() if r['imageable'])

        return {
            'baseline_km': baseline / 1e3,
            'wavelength_nm': wavelength * 1e9,
            'scenarios': results,
            'num_scenarios': len(scenarios),
            'num_resolvable': num_resolvable,
            'num_imageable': num_imageable,
            'success_rate_resolve': num_resolvable / len(scenarios),
            'success_rate_image': num_imageable / len(scenarios)
        }


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("ANGULAR RESOLUTION VALIDATION")
    print("=" * 60)

    # ===== VALIDATE PAPER CLAIM =====
    print("\n" + "-" * 60)
    print("PAPER CLAIM VALIDATION")
    print("-" * 60)

    validator = TransPlanckianResolutionValidator()
    validation = validator.validate_paper_claim()

    print(f"\nConfiguration:")
    print(f"  Wavelength: {validation['wavelength_nm']:.0f} nm")
    print(f"  Baseline: {validation['baseline_km']:.0f} km")

    print(f"\nResolution:")
    print(f"  Paper claim: {validation['paper_claim_microarcsec']:.2e} μas")
    print(f"  Calculated: {validation['calculated_microarcsec']:.2e} μas")
    print(f"  Ratio: {validation['ratio']:.2f}")
    print(f"  Agreement: {validation['agreement']}")

    metrics = validation['metrics']
    print(f"\nDetailed Metrics:")
    print(f"  θ = {metrics.angular_resolution_rad:.2e} rad")
    print(f"  θ = {metrics.angular_resolution_arcsec:.2e} arcsec")
    print(f"  θ = {metrics.angular_resolution_microarcsec:.2e} μas")

    # ===== COMPARE WITH EXISTING INSTRUMENTS =====
    print("\n" + "-" * 60)
    print("COMPARISON WITH EXISTING INSTRUMENTS")
    print("-" * 60)

    comparison = validator.compare_with_existing_instruments()

    print(f"\nTrans-Planckian resolution: {comparison['trans_planckian_resolution_microarcsec']:.2e} μas")
    print(f"\nExisting instruments:")
    for name, res in comparison['instrument_resolutions'].items():
        improvement = comparison['improvement_factors'][name]
        print(f"  {name:10s}: {res:.2e} μas (improvement: {improvement:.2e}×)")

    # ===== EXOPLANET IMAGING CAPABILITY =====
    print("\n" + "-" * 60)
    print("EXOPLANET IMAGING CAPABILITY")
    print("-" * 60)

    survey = validator.exoplanet_survey_capability()

    print(f"\nSurvey Configuration:")
    print(f"  Baseline: {survey['baseline_km']:.0f} km")
    print(f"  Wavelength: {survey['wavelength_nm']:.0f} nm")

    print(f"\nScenarios:")
    for name, result in survey['scenarios'].items():
        print(f"\n  {name}:")
        print(f"    Distance: {result['planet_distance_pc']:.0f} pc")
        print(f"    Angular size: {result['planet_angular_size_microarcsec']:.2e} μas")
        print(f"    Resolution: {result['resolution_microarcsec']:.2e} μas")
        print(f"    Spatial resolution: {result['spatial_resolution_km']:.1f} km")
        print(f"    Resolution elements: {result['resolution_elements_across_planet']:.1f}")
        print(f"    Resolvable: {result['resolvable']}")
        print(f"    Imageable: {result['imageable']}")

    print(f"\nSummary:")
    print(f"  Total scenarios: {survey['num_scenarios']}")
    print(f"  Resolvable: {survey['num_resolvable']} ({survey['success_rate_resolve']*100:.0f}%)")
    print(f"  Imageable: {survey['num_imageable']} ({survey['success_rate_image']*100:.0f}%)")

    # ===== PLOT RESOLUTION VS BASELINE =====
    print("\n" + "-" * 60)
    print("GENERATING PLOTS")
    print("-" * 60)

    wavelengths = [500e-9, 1e-6, 10e-6]  # Visible, NIR, MIR
    baselines = np.logspace(0, 8, 100)  # 1 m to 100,000 km

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Resolution vs baseline
    for wl in wavelengths:
        calc = AngularResolutionCalculator(wl)
        resolutions = [calc.angular_resolution_microarcsec(b) for b in baselines]
        axes[0].loglog(baselines/1e3, resolutions,
                      label=f'λ = {wl*1e9:.0f} nm', linewidth=2)

    # Mark paper claim
    axes[0].axhline(1e-5, color='r', linestyle='--', linewidth=2,
                   label='Paper claim (10⁻⁵ μas)')
    axes[0].axvline(1e4, color='r', linestyle='--', linewidth=2, alpha=0.5)

    axes[0].set_xlabel('Baseline [km]')
    axes[0].set_ylabel('Angular Resolution [μas]')
    axes[0].set_title('Angular Resolution vs Baseline')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which='both')

    # Exoplanet detection limits
    calc = AngularResolutionCalculator(500e-9)
    distances_pc = [1, 5, 10, 50, 100]

    for dist in distances_pc:
        planet_sizes = []
        for b in baselines:
            # Minimum detectable planet size (1 resolution element)
            spatial_res = calc.spatial_resolution_at_distance(b, dist * 3.086e16)
            planet_sizes.append(spatial_res / 6.371e6)  # In Earth radii

        axes[1].loglog(baselines/1e3, planet_sizes,
                      label=f'{dist} pc', linewidth=2)

    axes[1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Earth size')
    axes[1].set_xlabel('Baseline [km]')
    axes[1].set_ylabel('Minimum Detectable Planet Size [Earth radii]')
    axes[1].set_title('Exoplanet Detection Limits')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('angular_resolution_validation.png', dpi=150)
    print("Plot saved: angular_resolution_validation.png")

    print("\n" + "=" * 60)
