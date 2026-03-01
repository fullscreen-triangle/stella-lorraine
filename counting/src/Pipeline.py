"""
State Counting Pipeline
=======================

Main orchestration module that ties together:
- mzML spectral reading (SpectraReader)
- Hardware oscillator state counting (TrappedIon)
- S-entropy transformation (EntropyTransformation)
- Thermodynamic regime classification (ThermodynamicRegimes)
- Ionization physics (IonisationPhysics)
- Dissociation modeling (Dissociation)
- Multi-modal detection (Detector)

This pipeline validates three theoretical frameworks:
1. Trans-Planckian: Bounded, discrete phase space
2. CatScript: Categorical partition coordinates
3. Categorical Cryogenics: T = 2E/(3k_B × M)

Author: Kundai Sachikonye
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import numpy as np

# Local imports
from .SpectraReader import extract_mzml, ppm_window_bounds, find_peaks_in_window
from .TrappedIon import (
    HardwareOscillator, IonTrajectory, IonState, PartitionCoordinates,
    create_ion_trajectory, JourneyStage, ThermodynamicRegime as TIRegime
)
from .ThermodynamicRegimes import (
    ThermodynamicRegimeClassifier, SEntropyCoordinates,
    ThermodynamicState, UniversalEquationOfState, RegimeTransitionDetector,
    calculate_categorical_temperature, ThermodynamicRegime
)
from .EntropyTransformation import SEntropyTransformer
from .IonisationPhysics import IonizationEngine
# from .Dissociation import DissociationEngine  # Not used in this module

logger = logging.getLogger(__name__)


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

K_B = 1.380649e-23          # Boltzmann constant (J/K)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the state counting pipeline."""

    # Oscillator settings
    oscillator_frequency_hz: float = 10e6  # 10 MHz quartz
    oscillator_stability: float = 1e-9

    # Instrument type
    instrument_type: str = "orbitrap"  # "orbitrap", "fticr", "tof"

    # mzML processing
    rt_range: List[float] = field(default_factory=lambda: [0.0, 60.0])
    dda_top: int = 10
    ms1_threshold: int = 1000
    ms2_threshold: int = 10
    ms1_precision: float = 5e-6   # 5 ppm
    ms2_precision: float = 20e-6  # 20 ppm
    vendor: str = "thermo"

    # Ion selection
    min_intensity: float = 1000.0
    max_charge: int = 10
    ppm_tolerance: float = 10.0

    # Energy assumptions (can be refined)
    default_kinetic_energy_eV: float = 10.0
    collision_energy_eV: float = 30.0

    # Validation
    validate_trans_planckian: bool = True
    validate_catscript: bool = True
    validate_categorical_cryogenics: bool = True

    # Output
    output_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'oscillator_frequency_hz': self.oscillator_frequency_hz,
            'instrument_type': self.instrument_type,
            'rt_range': self.rt_range,
            'dda_top': self.dda_top,
            'ms1_threshold': self.ms1_threshold,
            'ms2_threshold': self.ms2_threshold,
            'vendor': self.vendor,
            'min_intensity': self.min_intensity,
            'ppm_tolerance': self.ppm_tolerance,
            'default_kinetic_energy_eV': self.default_kinetic_energy_eV
        }


# ============================================================================
# ION RECORD
# ============================================================================

@dataclass
class IonRecord:
    """
    Complete record for a single ion processed through the pipeline.
    """
    # Identity
    ion_id: int
    mz: float
    charge: int
    intensity: float
    retention_time: float

    # Trajectory
    trajectory: Optional[IonTrajectory] = None

    # State counting
    total_state_count: int = 0
    partition_coords: Optional[PartitionCoordinates] = None

    # S-Entropy
    s_entropy: Optional[SEntropyCoordinates] = None

    # Thermodynamic state
    thermo_state: Optional[ThermodynamicState] = None
    regime: Optional[ThermodynamicRegime] = None

    # Validation results
    validation: Dict[str, Any] = field(default_factory=dict)

    # MS2 data (if available)
    has_ms2: bool = False
    fragments: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ion_id': self.ion_id,
            'mz': self.mz,
            'charge': self.charge,
            'intensity': self.intensity,
            'retention_time': self.retention_time,
            'total_state_count': self.total_state_count,
            'partition': self.partition_coords.to_dict() if self.partition_coords else None,
            's_entropy': self.s_entropy.to_dict() if self.s_entropy else None,
            'regime': self.regime.name if self.regime else None,
            'has_ms2': self.has_ms2,
            'n_fragments': len(self.fragments),
            'validation': self.validation
        }


# ============================================================================
# PIPELINE RESULTS
# ============================================================================

@dataclass
class PipelineResults:
    """Results from processing an mzML file."""

    # Metadata
    input_file: str
    processing_time: datetime
    config: PipelineConfig

    # Ion records
    ions: List[IonRecord] = field(default_factory=list)

    # Summary statistics
    n_ions_processed: int = 0
    n_ms1_scans: int = 0
    n_ms2_scans: int = 0

    # Regime distribution
    regime_counts: Dict[str, int] = field(default_factory=dict)

    # Aggregate validation
    trans_planckian_validated: bool = False
    catscript_validated: bool = False
    categorical_cryogenics_validated: bool = False

    # Transition entropy
    transitions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'metadata': {
                'input_file': self.input_file,
                'processing_time': self.processing_time.isoformat(),
                'config': self.config.to_dict()
            },
            'summary': {
                'n_ions_processed': self.n_ions_processed,
                'n_ms1_scans': self.n_ms1_scans,
                'n_ms2_scans': self.n_ms2_scans,
                'regime_distribution': self.regime_counts
            },
            'validation': {
                'trans_planckian': self.trans_planckian_validated,
                'catscript': self.catscript_validated,
                'categorical_cryogenics': self.categorical_cryogenics_validated
            },
            'ions': [ion.to_dict() for ion in self.ions[:100]],  # Limit for JSON
            'transitions': self.transitions
        }

    def save(self, output_path: str):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")


# ============================================================================
# STATE COUNTING PIPELINE
# ============================================================================

class StateCountingPipeline:
    """
    Main pipeline for processing mass spectrometry data through
    the state counting framework.

    The pipeline validates:
    1. Trans-Planckian: Phase space is bounded and discrete
    2. CatScript: Partition coordinates derived from counts
    3. Categorical Cryogenics: T = 2E / (3k_B × M)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self.oscillator = HardwareOscillator(
            frequency_hz=self.config.oscillator_frequency_hz,
            stability=self.config.oscillator_stability
        )

        self.regime_classifier = ThermodynamicRegimeClassifier()
        self.transition_detector = RegimeTransitionDetector()
        self.eos = UniversalEquationOfState()

        # Processing state
        self.current_results: Optional[PipelineResults] = None
        self._ion_counter = 0

    def process_mzml(self, mzml_path: str) -> PipelineResults:
        """
        Process an mzML file through the state counting pipeline.

        Args:
            mzml_path: Path to mzML file

        Returns:
            PipelineResults object
        """
        logger.info(f"Processing: {mzml_path}")
        start_time = datetime.now()

        # Initialize results
        self.current_results = PipelineResults(
            input_file=mzml_path,
            processing_time=start_time,
            config=self.config
        )

        # Extract spectra from mzML
        try:
            scan_info_df, spectra_dict, ms1_xic_df = extract_mzml(
                mzml_path,
                rt_range=self.config.rt_range,
                dda_top=self.config.dda_top,
                ms1_threshold=self.config.ms1_threshold,
                ms2_threshold=self.config.ms2_threshold,
                ms1_precision=self.config.ms1_precision,
                ms2_precision=self.config.ms2_precision,
                vendor=self.config.vendor
            )
        except Exception as e:
            logger.error(f"Failed to extract mzML: {e}")
            raise

        # Count scans
        self.current_results.n_ms1_scans = len(
            scan_info_df[scan_info_df['DDA_rank'] == 0]
        )
        self.current_results.n_ms2_scans = len(
            scan_info_df[scan_info_df['DDA_rank'] > 0]
        )

        # Process MS1 peaks
        self._process_ms1_peaks(ms1_xic_df, spectra_dict, scan_info_df)

        # Calculate aggregate validations
        self._calculate_aggregate_validation()

        # Get regime distribution
        for ion in self.current_results.ions:
            if ion.regime:
                regime_name = ion.regime.name
                self.current_results.regime_counts[regime_name] = \
                    self.current_results.regime_counts.get(regime_name, 0) + 1

        # Get transitions
        self.current_results.transitions = self.transition_detector.get_all_transitions()

        logger.info(f"Processed {self.current_results.n_ions_processed} ions")
        return self.current_results

    def _process_ms1_peaks(
        self,
        ms1_xic_df,
        spectra_dict: Dict,
        scan_info_df
    ):
        """Process MS1 peaks from XIC data."""
        if ms1_xic_df.empty:
            logger.warning("No MS1 data to process")
            return

        # Get unique m/z values above intensity threshold
        high_intensity_peaks = ms1_xic_df[
            ms1_xic_df['i'] >= self.config.min_intensity
        ]

        # Group by rounded m/z to avoid processing isotopes separately
        unique_mz = high_intensity_peaks['mz'].round(1).unique()

        logger.info(f"Processing {len(unique_mz)} unique m/z values")

        for mz_rounded in unique_mz:
            # Get representative peak
            mask = (ms1_xic_df['mz'].round(1) == mz_rounded)
            peaks = ms1_xic_df[mask]

            if peaks.empty:
                continue

            # Use most intense occurrence
            best_idx = peaks['i'].idxmax()
            best_peak = peaks.loc[best_idx]

            ion = self._process_single_ion(
                mz=float(best_peak['mz']),
                intensity=float(best_peak['i']),
                rt=float(best_peak['rt']),
                spectra_dict=spectra_dict,
                scan_info_df=scan_info_df
            )

            if ion:
                self.current_results.ions.append(ion)
                self.current_results.n_ions_processed += 1

    def _process_single_ion(
        self,
        mz: float,
        intensity: float,
        rt: float,
        spectra_dict: Dict,
        scan_info_df
    ) -> Optional[IonRecord]:
        """
        Process a single ion through the state counting framework.

        Args:
            mz: m/z value
            intensity: Peak intensity
            rt: Retention time
            spectra_dict: Dictionary of spectra
            scan_info_df: Scan information DataFrame

        Returns:
            IonRecord or None if processing fails
        """
        self._ion_counter += 1
        charge = self._estimate_charge(mz)

        # Create ion trajectory
        trajectory = create_ion_trajectory(
            mz=mz,
            intensity=intensity,
            charge=charge,
            energy_eV=self.config.default_kinetic_energy_eV,
            oscillator_freq_hz=self.config.oscillator_frequency_hz,
            instrument=self.config.instrument_type
        )

        # Complete MS1 journey
        trajectory.complete_ms1_journey()

        # Get state count
        total_M = trajectory.get_total_count()

        # Get partition coordinates from final state
        final_state = trajectory.get_final_state()
        partition = final_state.partition if final_state else None

        # Calculate S-entropy coordinates
        s_entropy = self._calculate_s_entropy(mz, intensity, rt, total_M)

        # Classify regime
        regime, params = self.regime_classifier.classify(
            mz, charge, self.config.default_kinetic_energy_eV, total_M
        )

        # Check for regime transition
        self.transition_detector.check_transition(
            mz, charge, self.config.default_kinetic_energy_eV, total_M,
            stage_name=f"ion_{self._ion_counter}"
        )

        # Create thermodynamic state
        thermo_state = ThermodynamicState(
            n=partition.n if partition else 1,
            l=partition.l if partition else 0,
            m=partition.m if partition else 0,
            s=partition.s if partition else 0.5,
            M=total_M,
            s_entropy=s_entropy,
            params=params,
            temperature_K=300.0,  # Ambient
            energy_eV=self.config.default_kinetic_energy_eV,
            mz=mz,
            charge=charge,
            regime=regime
        )

        # Get validation report
        validation = trajectory.get_validation_report()

        # Create ion record
        ion = IonRecord(
            ion_id=self._ion_counter,
            mz=mz,
            charge=charge,
            intensity=intensity,
            retention_time=rt,
            trajectory=trajectory,
            total_state_count=total_M,
            partition_coords=partition,
            s_entropy=s_entropy,
            thermo_state=thermo_state,
            regime=regime,
            validation=validation
        )

        # Check for MS2 data
        ion.has_ms2, ion.fragments = self._find_ms2_data(
            mz, scan_info_df, spectra_dict
        )

        return ion

    def _estimate_charge(self, mz: float) -> int:
        """Estimate charge state from m/z."""
        # Simple heuristic - could be improved with isotope pattern analysis
        if mz < 300:
            return 1
        elif mz < 600:
            return 1
        elif mz < 1000:
            return 2
        elif mz < 1500:
            return 3
        else:
            return min(int(mz / 400), self.config.max_charge)

    def _calculate_s_entropy(
        self,
        mz: float,
        intensity: float,
        rt: float,
        state_count: int
    ) -> SEntropyCoordinates:
        """
        Calculate S-entropy coordinates for an ion.

        S_k: Knowledge entropy - based on peak intensity (higher = more certain)
        S_t: Temporal entropy - based on retention time spread
        S_e: Energy entropy - based on state count
        """
        # Normalize intensity to [0, 1]
        s_k = min(1.0, intensity / 1e7)  # Saturates at 10^7

        # Temporal entropy - inversely related to state count
        s_t = 1.0 / np.sqrt(max(1, state_count)) if state_count > 0 else 1.0

        # Energy entropy - based on m/z distribution
        s_e = min(1.0, np.log10(max(1, mz)) / 4)  # Saturates around m/z 10000

        return SEntropyCoordinates(s_k=s_k, s_t=s_t, s_e=s_e)

    def _find_ms2_data(
        self,
        precursor_mz: float,
        scan_info_df,
        spectra_dict: Dict
    ) -> Tuple[bool, List[Dict[str, float]]]:
        """Find MS2 data for a precursor."""
        # Handle synthetic data (no scan info)
        if scan_info_df is None or spectra_dict is None:
            return False, []

        ppm_tol = self.config.ppm_tolerance
        lower, upper = ppm_window_bounds(precursor_mz, ppm_tol)

        # Find matching MS2 scans
        ms2_scans = scan_info_df[
            (scan_info_df['DDA_rank'] > 0) &
            (scan_info_df['MS2_PR_mz'] >= lower) &
            (scan_info_df['MS2_PR_mz'] <= upper)
        ]

        if ms2_scans.empty:
            return False, []

        # Get fragments from first matching scan
        first_scan_idx = ms2_scans['spec_index'].iloc[0]
        if first_scan_idx not in spectra_dict:
            return True, []

        spec_df = spectra_dict[first_scan_idx]
        fragments = [
            {'mz': float(row['mz']), 'intensity': float(row['i'])}
            for _, row in spec_df.iterrows()
        ]

        return True, fragments

    def _calculate_aggregate_validation(self):
        """Calculate aggregate validation results."""
        if not self.current_results or not self.current_results.ions:
            return

        # Trans-Planckian: Check all ions have bounded states
        trans_planckian_valid = all(
            ion.validation.get('trans_planckian', {}).get('bounded', False)
            for ion in self.current_results.ions
            if ion.validation
        )
        self.current_results.trans_planckian_validated = trans_planckian_valid

        # CatScript: Check all partition coordinates are valid
        catscript_valid = all(
            ion.partition_coords.is_valid() if ion.partition_coords else False
            for ion in self.current_results.ions
        )
        self.current_results.catscript_validated = catscript_valid

        # Categorical Cryogenics: Check temperature suppression
        # Valid if T_cat < T_classical for all ions
        cc_valid = all(
            ion.thermo_state.temperature_suppression < 1.0
            if ion.thermo_state else False
            for ion in self.current_results.ions
        )
        self.current_results.categorical_cryogenics_validated = cc_valid

    def process_peak_list(
        self,
        peaks: List[Dict[str, float]]
    ) -> PipelineResults:
        """
        Process a list of peaks (alternative to mzML).

        Args:
            peaks: List of dicts with 'mz', 'intensity', optional 'rt'

        Returns:
            PipelineResults object
        """
        start_time = datetime.now()

        self.current_results = PipelineResults(
            input_file="peak_list",
            processing_time=start_time,
            config=self.config
        )

        for peak in peaks:
            mz = peak.get('mz', 500.0)
            intensity = peak.get('intensity', 1000.0)
            rt = peak.get('rt', 0.0)

            ion = self._process_single_ion(
                mz=mz,
                intensity=intensity,
                rt=rt,
                spectra_dict={},
                scan_info_df=None
            )

            if ion:
                self.current_results.ions.append(ion)
                self.current_results.n_ions_processed += 1

        self._calculate_aggregate_validation()
        return self.current_results


# ============================================================================
# VALIDATION PIPELINE
# ============================================================================

class ValidationPipeline:
    """
    Dedicated pipeline for validating theoretical claims
    against mass spectrometry data.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.pipeline = StateCountingPipeline(config)

    def validate_trans_planckian(
        self,
        results: PipelineResults
    ) -> Dict[str, Any]:
        """
        Validate trans-Planckian claims:
        1. Phase space is bounded
        2. States are discrete (countable)
        3. No UV divergence (capacity formula)
        """
        validations = []

        for ion in results.ions:
            if not ion.partition_coords:
                continue

            n = ion.partition_coords.n
            capacity = ion.partition_coords.capacity
            cumulative = ion.partition_coords.cumulative_capacity
            M = ion.total_state_count

            validations.append({
                'ion_id': ion.ion_id,
                'mz': ion.mz,
                'n': n,
                'capacity': capacity,
                'cumulative_capacity': cumulative,
                'state_count_M': M,
                'bounded': M <= cumulative,
                'discrete': isinstance(M, int)
            })

        all_bounded = all(v['bounded'] for v in validations)
        all_discrete = all(v['discrete'] for v in validations)

        return {
            'claim': 'Phase space is bounded and discrete',
            'validations': validations[:50],  # Limit output
            'n_ions': len(validations),
            'all_bounded': all_bounded,
            'all_discrete': all_discrete,
            'overall_valid': all_bounded and all_discrete
        }

    def validate_catscript(
        self,
        results: PipelineResults
    ) -> Dict[str, Any]:
        """
        Validate CatScript claims:
        1. Partition coordinates derived from oscillator counts
        2. Coordinates satisfy selection rules
        3. State index is unique
        """
        validations = []

        for ion in results.ions:
            if not ion.partition_coords:
                continue

            p = ion.partition_coords
            M = ion.total_state_count

            # Verify n derived from M: n ≈ sqrt(M/2) + 1
            expected_n = max(1, int(np.sqrt(M / 2)) + 1)

            validations.append({
                'ion_id': ion.ion_id,
                'M': M,
                'n': p.n,
                'expected_n': expected_n,
                'n_correct': abs(p.n - expected_n) <= 1,
                'l_valid': 0 <= p.l < p.n,
                'm_valid': -p.l <= p.m <= p.l,
                's_valid': p.s in [-0.5, 0.5],
                'state_index': p.state_index
            })

        all_valid = all(
            v['n_correct'] and v['l_valid'] and v['m_valid'] and v['s_valid']
            for v in validations
        )

        return {
            'claim': 'Partition coordinates from oscillator counts',
            'capacity_formula': 'C(n) = 2n^2',
            'validations': validations[:50],
            'n_ions': len(validations),
            'overall_valid': all_valid
        }

    def validate_categorical_cryogenics(
        self,
        results: PipelineResults
    ) -> Dict[str, Any]:
        """
        Validate categorical cryogenics claims:
        1. T_cat = 2E / (3k_B × M)
        2. Temperature suppression = 1/M
        3. More states → Lower temperature
        """
        validations = []

        for ion in results.ions:
            if not ion.thermo_state:
                continue

            M = ion.total_state_count
            E_eV = self.config.default_kinetic_energy_eV
            E_J = E_eV * E_CHARGE

            # Expected categorical temperature
            T_expected = 2 * E_J / (3 * K_B * max(1, M))

            # Actual from state
            T_actual = ion.thermo_state.categorical_temperature_K

            validations.append({
                'ion_id': ion.ion_id,
                'M': M,
                'E_eV': E_eV,
                'T_categorical': T_actual,
                'T_expected': T_expected,
                'T_match': abs(T_actual - T_expected) / max(T_expected, 1e-10) < 0.01,
                'suppression': ion.thermo_state.temperature_suppression,
                'expected_suppression': 1.0 / max(1, M)
            })

        all_match = all(v['T_match'] for v in validations)

        return {
            'claim': 'T = 2E / (3k_B × M)',
            'insight': 'More states → Lower effective temperature',
            'validations': validations[:50],
            'n_ions': len(validations),
            'overall_valid': all_match
        }

    def validate_all(
        self,
        results: PipelineResults
    ) -> Dict[str, Any]:
        """Run all validations."""
        return {
            'trans_planckian': self.validate_trans_planckian(results),
            'catscript': self.validate_catscript(results),
            'categorical_cryogenics': self.validate_categorical_cryogenics(results),
            'summary': {
                'n_ions': results.n_ions_processed,
                'regime_distribution': results.regime_counts
            }
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def process_mzml_file(
    mzml_path: str,
    output_path: Optional[str] = None,
    **config_kwargs
) -> PipelineResults:
    """
    Process an mzML file with default configuration.

    Args:
        mzml_path: Path to mzML file
        output_path: Optional path to save results JSON
        **config_kwargs: Override config parameters

    Returns:
        PipelineResults object
    """
    config = PipelineConfig(**config_kwargs)
    pipeline = StateCountingPipeline(config)

    results = pipeline.process_mzml(mzml_path)

    if output_path:
        results.save(output_path)

    return results


def validate_from_mzml(
    mzml_path: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process mzML and run all validations.

    Args:
        mzml_path: Path to mzML file
        output_path: Optional path to save validation results

    Returns:
        Validation results dictionary
    """
    pipeline = StateCountingPipeline()
    results = pipeline.process_mzml(mzml_path)

    validator = ValidationPipeline()
    validation = validator.validate_all(results)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(validation, f, indent=2)

    return validation


def demonstrate_pipeline():
    """Demonstrate pipeline with synthetic data."""
    print("=" * 70)
    print("STATE COUNTING PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Create pipeline
    config = PipelineConfig(
        oscillator_frequency_hz=10e6,
        instrument_type="orbitrap",
        default_kinetic_energy_eV=10.0
    )
    pipeline = StateCountingPipeline(config)

    # Process synthetic peak list
    test_peaks = [
        {'mz': 200.0, 'intensity': 5000, 'rt': 5.0},
        {'mz': 350.5, 'intensity': 12000, 'rt': 7.5},
        {'mz': 500.25, 'intensity': 8000, 'rt': 10.0},
        {'mz': 750.3, 'intensity': 3000, 'rt': 12.5},
        {'mz': 1000.0, 'intensity': 15000, 'rt': 15.0},
    ]

    results = pipeline.process_peak_list(test_peaks)

    print(f"\nProcessed {results.n_ions_processed} ions")
    print(f"\nRegime distribution:")
    for regime, count in results.regime_counts.items():
        print(f"  {regime}: {count}")

    print(f"\nValidation Summary:")
    print(f"  Trans-Planckian: {'PASS' if results.trans_planckian_validated else 'FAIL'}")
    print(f"  CatScript: {'PASS' if results.catscript_validated else 'FAIL'}")
    print(f"  Categorical Cryogenics: {'PASS' if results.categorical_cryogenics_validated else 'FAIL'}")

    # Run detailed validation
    validator = ValidationPipeline()
    validation = validator.validate_all(results)

    print(f"\nDetailed Validation:")
    for framework, result in validation.items():
        if isinstance(result, dict) and 'overall_valid' in result:
            status = 'PASS' if result['overall_valid'] else 'FAIL'
            print(f"  {framework}: {status}")

    print("\n" + "=" * 70)
    print("Sample Ion Records:")
    print("=" * 70)

    for ion in results.ions[:3]:
        print(f"\n  Ion {ion.ion_id}: m/z = {ion.mz:.2f}")
        print(f"    State count M = {ion.total_state_count:,}")
        print(f"    Partition: n={ion.partition_coords.n}, l={ion.partition_coords.l}")
        print(f"    Regime: {ion.regime.name if ion.regime else 'N/A'}")
        print(f"    T_categorical = {ion.thermo_state.categorical_temperature_K:.2e} K")


if __name__ == "__main__":
    demonstrate_pipeline()
