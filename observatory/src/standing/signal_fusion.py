"""
Signal Fusion - Sophisticated Time Signal Combination

This module implements sophisticated methods for combining time signals of different
precisions using Kalman filtering, weighted least squares, and other appropriate
statistical fusion techniques for optimal temporal accuracy.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from scipy import linalg
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings('ignore')


class FusionAlgorithm(Enum):
    """Signal fusion algorithms"""
    KALMAN_FILTER = "kalman_filter"
    EXTENDED_KALMAN = "extended_kalman"
    WEIGHTED_LEAST_SQUARES = "weighted_least_squares"
    PARTICLE_FILTER = "particle_filter"
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    BAYESIAN_FUSION = "bayesian_fusion"
    ADAPTIVE_FUSION = "adaptive_fusion"


class SignalType(Enum):
    """Types of time signals"""
    ATOMIC_CLOCK = "atomic_clock"
    GPS_TIME = "gps_time"
    NTP_TIME = "ntp_time"
    CELLULAR_TIME = "cellular_time"
    VIRTUAL_REFERENCE = "virtual_reference"
    SYSTEM_CLOCK = "system_clock"


class FusionQuality(Enum):
    """Quality levels of fused signals"""
    EXCELLENT = "excellent"      # < 1ns uncertainty
    VERY_GOOD = "very_good"      # < 10ns uncertainty
    GOOD = "good"                # < 100ns uncertainty
    ACCEPTABLE = "acceptable"    # < 1μs uncertainty
    POOR = "poor"               # > 1μs uncertainty


@dataclass
class TimeSignal:
    """Individual time signal for fusion"""
    signal_id: str
    signal_type: SignalType
    timestamp: float
    precision: float           # Standard deviation in seconds
    confidence: float          # Confidence level (0-1)
    latency: float            # Signal latency in seconds
    drift_rate: float         # Clock drift rate (s/s)
    quality_factor: float     # Overall quality metric
    measurement_time: float   # When this measurement was taken
    source_info: Dict = field(default_factory=dict)

    def get_uncertainty(self) -> float:
        """Calculate total uncertainty including precision and latency effects"""
        latency_uncertainty = self.latency * abs(self.drift_rate) if self.drift_rate != 0 else 0
        total_uncertainty = math.sqrt(self.precision**2 + latency_uncertainty**2)
        return total_uncertainty

    def get_weight(self) -> float:
        """Calculate weight for fusion based on uncertainty and confidence"""
        uncertainty = self.get_uncertainty()
        if uncertainty == 0:
            return 1e6  # Very high weight for perfect signals
        base_weight = 1.0 / uncertainty**2
        confidence_weight = base_weight * self.confidence * self.quality_factor
        return confidence_weight


@dataclass
class FusionResult:
    """Result of signal fusion process"""
    fused_timestamp: float
    fused_precision: float
    fusion_confidence: float
    fusion_quality: FusionQuality
    input_signals_count: int
    fusion_algorithm: FusionAlgorithm
    processing_time: float
    fusion_metadata: Dict = field(default_factory=dict)

    def get_precision_improvement_factor(self, best_input_precision: float) -> float:
        """Calculate precision improvement factor over best input signal"""
        if best_input_precision == 0:
            return float('inf')
        return best_input_precision / self.fused_precision


@dataclass
class KalmanState:
    """Kalman filter state for time signal fusion"""
    state_estimate: np.ndarray      # [time, drift_rate]
    covariance_matrix: np.ndarray   # State covariance
    process_noise: np.ndarray       # Process noise covariance
    measurement_noise: float        # Measurement noise variance
    last_update_time: float

    def predict(self, dt: float):
        """Predict next state"""
        # State transition matrix: [1, dt; 0, 1]
        F = np.array([[1.0, dt], [0.0, 1.0]])

        # Predict state
        self.state_estimate = F @ self.state_estimate

        # Predict covariance
        self.covariance_matrix = F @ self.covariance_matrix @ F.T + self.process_noise

    def update(self, measurement: float, measurement_variance: float):
        """Update state with new measurement"""
        # Measurement matrix: [1, 0] (we measure time directly)
        H = np.array([[1.0, 0.0]])

        # Innovation
        innovation = measurement - H @ self.state_estimate

        # Innovation covariance
        S = H @ self.covariance_matrix @ H.T + measurement_variance

        # Kalman gain
        K = self.covariance_matrix @ H.T / S

        # Update state estimate
        self.state_estimate = self.state_estimate + K * innovation

        # Update covariance
        I = np.eye(2)
        self.covariance_matrix = (I - K @ H) @ self.covariance_matrix


class SignalFusionEngine:
    """
    Signal Fusion Engine for Multi-Precision Time Signal Combination

    Implements sophisticated fusion algorithms to combine time signals from multiple
    sources with different precisions, latencies, and quality characteristics.

    Key Features:
    - Kalman filtering for optimal statistical fusion
    - Weighted least squares for robust estimation
    - Adaptive algorithms that adjust to signal characteristics
    - Real-time fusion with minimal computational overhead
    - Precision improvement through intelligent signal weighting
    """

    def __init__(self):
        self.fusion_history: List[FusionResult] = []
        self.kalman_states: Dict[str, KalmanState] = {}
        self.signal_buffer: Dict[str, List[TimeSignal]] = {}

        # Fusion parameters
        self.buffer_size = 100
        self.minimum_signals_for_fusion = 2
        self.maximum_fusion_latency = 1.0  # seconds

        # Algorithm-specific parameters
        self.kalman_config = {
            'process_noise_std': 1e-9,      # 1 nanosecond process noise
            'initial_drift_uncertainty': 1e-6,  # 1 ppm initial drift uncertainty
            'initial_time_uncertainty': 1e-3     # 1 ms initial time uncertainty
        }

        # Quality thresholds
        self.quality_thresholds = {
            FusionQuality.EXCELLENT: 1e-9,      # 1 nanosecond
            FusionQuality.VERY_GOOD: 1e-8,      # 10 nanoseconds
            FusionQuality.GOOD: 1e-7,           # 100 nanoseconds
            FusionQuality.ACCEPTABLE: 1e-6,     # 1 microsecond
            FusionQuality.POOR: float('inf')    # Worse than 1 microsecond
        }

        # Performance tracking
        self.performance_stats = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'average_precision_improvement': 0.0,
            'average_processing_time': 0.0
        }

    def add_time_signal(self, signal: TimeSignal):
        """Add time signal to fusion buffer"""

        # Initialize buffer for this signal type if needed
        if signal.signal_id not in self.signal_buffer:
            self.signal_buffer[signal.signal_id] = []

        # Add signal to buffer
        self.signal_buffer[signal.signal_id].append(signal)

        # Maintain buffer size
        if len(self.signal_buffer[signal.signal_id]) > self.buffer_size:
            self.signal_buffer[signal.signal_id] = self.signal_buffer[signal.signal_id][-self.buffer_size:]

    def fuse_signals(self,
                    signals: List[TimeSignal],
                    algorithm: FusionAlgorithm = FusionAlgorithm.KALMAN_FILTER) -> FusionResult:
        """Fuse multiple time signals using specified algorithm"""

        start_time = time.time()

        if len(signals) < self.minimum_signals_for_fusion:
            return FusionResult(
                fused_timestamp=signals[0].timestamp if signals else time.time(),
                fused_precision=signals[0].precision if signals else 1.0,
                fusion_confidence=0.0,
                fusion_quality=FusionQuality.POOR,
                input_signals_count=len(signals),
                fusion_algorithm=algorithm,
                processing_time=time.time() - start_time,
                fusion_metadata={'error': 'insufficient_signals'}
            )

        # Route to appropriate fusion algorithm
        if algorithm == FusionAlgorithm.KALMAN_FILTER:
            result = self._kalman_filter_fusion(signals)
        elif algorithm == FusionAlgorithm.EXTENDED_KALMAN:
            result = self._extended_kalman_fusion(signals)
        elif algorithm == FusionAlgorithm.WEIGHTED_LEAST_SQUARES:
            result = self._weighted_least_squares_fusion(signals)
        elif algorithm == FusionAlgorithm.PARTICLE_FILTER:
            result = self._particle_filter_fusion(signals)
        elif algorithm == FusionAlgorithm.MAXIMUM_LIKELIHOOD:
            result = self._maximum_likelihood_fusion(signals)
        elif algorithm == FusionAlgorithm.BAYESIAN_FUSION:
            result = self._bayesian_fusion(signals)
        elif algorithm == FusionAlgorithm.ADAPTIVE_FUSION:
            result = self._adaptive_fusion(signals)
        else:
            result = self._weighted_least_squares_fusion(signals)  # Default fallback

        result.processing_time = time.time() - start_time
        result.fusion_algorithm = algorithm
        result.input_signals_count = len(signals)

        # Determine fusion quality
        result.fusion_quality = self._assess_fusion_quality(result.fused_precision)

        # Update performance statistics
        self._update_performance_stats(result, signals)

        # Store result
        self.fusion_history.append(result)

        return result

    def _kalman_filter_fusion(self, signals: List[TimeSignal]) -> FusionResult:
        """Kalman filter-based signal fusion"""

        # Initialize Kalman state if needed
        fusion_id = "kalman_fusion"
        current_time = time.time()

        if fusion_id not in self.kalman_states:
            # Initialize with first signal
            initial_state = np.array([signals[0].timestamp, 0.0])  # [time, drift_rate]
            initial_covariance = np.diag([
                self.kalman_config['initial_time_uncertainty']**2,
                self.kalman_config['initial_drift_uncertainty']**2
            ])
            process_noise = np.diag([
                self.kalman_config['process_noise_std']**2,
                (self.kalman_config['process_noise_std'] / 100)**2
            ])

            self.kalman_states[fusion_id] = KalmanState(
                state_estimate=initial_state,
                covariance_matrix=initial_covariance,
                process_noise=process_noise,
                measurement_noise=signals[0].precision**2,
                last_update_time=current_time
            )

        kalman_state = self.kalman_states[fusion_id]

        # Predict step
        dt = current_time - kalman_state.last_update_time
        if dt > 0:
            kalman_state.predict(dt)

        # Update with each measurement
        for signal in signals:
            measurement_variance = signal.get_uncertainty()**2
            kalman_state.update(signal.timestamp, measurement_variance)

        kalman_state.last_update_time = current_time

        # Extract results
        fused_timestamp = float(kalman_state.state_estimate[0])
        fused_precision = math.sqrt(float(kalman_state.covariance_matrix[0, 0]))

        # Calculate fusion confidence based on covariance reduction
        initial_uncertainty = max(signal.get_uncertainty() for signal in signals)
        confidence = min(1.0, initial_uncertainty / fused_precision) if fused_precision > 0 else 0.5

        return FusionResult(
            fused_timestamp=fused_timestamp,
            fused_precision=fused_precision,
            fusion_confidence=confidence,
            fusion_quality=FusionQuality.GOOD,  # Will be updated by caller
            input_signals_count=len(signals),
            fusion_algorithm=FusionAlgorithm.KALMAN_FILTER,
            processing_time=0.0,  # Will be set by caller
            fusion_metadata={
                'kalman_state': kalman_state.state_estimate.tolist(),
                'covariance_trace': float(np.trace(kalman_state.covariance_matrix)),
                'predicted_drift_rate': float(kalman_state.state_estimate[1])
            }
        )

    def _extended_kalman_fusion(self, signals: List[TimeSignal]) -> FusionResult:
        """Extended Kalman filter for nonlinear signal characteristics"""

        # For time signals, EKF reduces to regular Kalman filter in most cases
        # This implementation adds nonlinear corrections for clock drift

        result = self._kalman_filter_fusion(signals)

        # Add nonlinear drift corrections
        drift_corrections = []
        for signal in signals:
            if signal.drift_rate != 0:
                # Nonlinear drift correction based on measurement age
                age = time.time() - signal.measurement_time
                drift_correction = signal.drift_rate * age**2 / 2  # Quadratic drift term
                drift_corrections.append(drift_correction)

        if drift_corrections:
            avg_drift_correction = np.mean(drift_corrections)
            result.fused_timestamp += avg_drift_correction
            result.fusion_metadata['nonlinear_drift_correction'] = avg_drift_correction

        result.fusion_algorithm = FusionAlgorithm.EXTENDED_KALMAN
        return result

    def _weighted_least_squares_fusion(self, signals: List[TimeSignal]) -> FusionResult:
        """Weighted least squares fusion"""

        # Calculate weights based on signal uncertainty and quality
        weights = np.array([signal.get_weight() for signal in signals])
        timestamps = np.array([signal.timestamp for signal in signals])

        # Weighted mean
        total_weight = np.sum(weights)
        if total_weight == 0:
            fused_timestamp = np.mean(timestamps)
            fused_precision = np.std(timestamps) if len(timestamps) > 1 else signals[0].precision
            confidence = 0.5
        else:
            fused_timestamp = np.sum(weights * timestamps) / total_weight

            # Weighted standard deviation
            weighted_variance = np.sum(weights * (timestamps - fused_timestamp)**2) / total_weight
            fused_precision = math.sqrt(weighted_variance / len(signals))  # Standard error of mean

            # Confidence based on weight concentration
            weight_entropy = -np.sum((weights / total_weight) * np.log(weights / total_weight + 1e-10))
            max_entropy = math.log(len(weights))
            confidence = 1.0 - (weight_entropy / max_entropy) if max_entropy > 0 else 0.5

        return FusionResult(
            fused_timestamp=float(fused_timestamp),
            fused_precision=float(fused_precision),
            fusion_confidence=float(confidence),
            fusion_quality=FusionQuality.GOOD,
            input_signals_count=len(signals),
            fusion_algorithm=FusionAlgorithm.WEIGHTED_LEAST_SQUARES,
            processing_time=0.0,
            fusion_metadata={
                'weights': weights.tolist(),
                'weight_entropy': float(weight_entropy) if 'weight_entropy' in locals() else 0.0,
                'effective_samples': float(total_weight / np.max(weights)) if np.max(weights) > 0 else 1.0
            }
        )

    def _particle_filter_fusion(self, signals: List[TimeSignal]) -> FusionResult:
        """Particle filter fusion for complex noise characteristics"""

        num_particles = 1000

        # Initialize particles around signal measurements
        particles = []
        for signal in signals:
            # Generate particles around each signal
            signal_particles = np.random.normal(
                signal.timestamp,
                signal.get_uncertainty(),
                num_particles // len(signals)
            )
            particles.extend(signal_particles)

        particles = np.array(particles)

        # Calculate particle weights based on likelihood
        weights = np.ones(len(particles))

        for i, particle in enumerate(particles):
            likelihood = 1.0
            for signal in signals:
                # Gaussian likelihood
                diff = particle - signal.timestamp
                likelihood *= np.exp(-0.5 * (diff / signal.get_uncertainty())**2)
            weights[i] = likelihood * signal.confidence

        # Normalize weights
        weights /= np.sum(weights)

        # Calculate weighted statistics
        fused_timestamp = np.sum(weights * particles)
        fused_variance = np.sum(weights * (particles - fused_timestamp)**2)
        fused_precision = math.sqrt(fused_variance)

        # Confidence based on effective sample size
        effective_sample_size = 1.0 / np.sum(weights**2)
        confidence = effective_sample_size / len(particles)

        return FusionResult(
            fused_timestamp=float(fused_timestamp),
            fused_precision=float(fused_precision),
            fusion_confidence=float(confidence),
            fusion_quality=FusionQuality.GOOD,
            input_signals_count=len(signals),
            fusion_algorithm=FusionAlgorithm.PARTICLE_FILTER,
            processing_time=0.0,
            fusion_metadata={
                'num_particles': len(particles),
                'effective_sample_size': float(effective_sample_size),
                'weight_concentration': float(np.max(weights))
            }
        )

    def _maximum_likelihood_fusion(self, signals: List[TimeSignal]) -> FusionResult:
        """Maximum likelihood estimation fusion"""

        def negative_log_likelihood(timestamp):
            """Negative log-likelihood function"""
            nll = 0.0
            for signal in signals:
                diff = timestamp - signal.timestamp
                variance = signal.get_uncertainty()**2
                nll += 0.5 * (diff**2 / variance + np.log(2 * np.pi * variance))
            return nll

        # Initial guess (weighted mean)
        weights = [signal.get_weight() for signal in signals]
        initial_guess = np.average([signal.timestamp for signal in signals], weights=weights)

        # Optimize likelihood
        try:
            from scipy.optimize import minimize_scalar
            result_opt = minimize_scalar(negative_log_likelihood, method='brent')
            fused_timestamp = result_opt.x

            # Calculate precision using Fisher information (Cramér-Rao bound)
            fisher_info = sum(1.0 / signal.get_uncertainty()**2 for signal in signals)
            fused_precision = 1.0 / math.sqrt(fisher_info) if fisher_info > 0 else np.mean([s.precision for s in signals])

            confidence = 0.9 if result_opt.success else 0.5

        except:
            # Fallback to weighted average
            fused_timestamp = initial_guess
            fused_precision = np.std([signal.timestamp for signal in signals])
            confidence = 0.5

        return FusionResult(
            fused_timestamp=float(fused_timestamp),
            fused_precision=float(fused_precision),
            fusion_confidence=float(confidence),
            fusion_quality=FusionQuality.GOOD,
            input_signals_count=len(signals),
            fusion_algorithm=FusionAlgorithm.MAXIMUM_LIKELIHOOD,
            processing_time=0.0,
            fusion_metadata={
                'optimization_success': confidence > 0.8,
                'fisher_information': float(fisher_info) if 'fisher_info' in locals() else 0.0
            }
        )

    def _bayesian_fusion(self, signals: List[TimeSignal]) -> FusionResult:
        """Bayesian fusion with prior information"""

        # Use system time as prior
        prior_mean = time.time()
        prior_variance = 1.0  # 1 second prior uncertainty

        # Bayesian update with each signal
        posterior_mean = prior_mean
        posterior_variance = prior_variance

        for signal in signals:
            # Likelihood from signal
            likelihood_mean = signal.timestamp
            likelihood_variance = signal.get_uncertainty()**2

            # Bayesian update
            precision_prior = 1.0 / posterior_variance
            precision_likelihood = 1.0 / likelihood_variance
            precision_posterior = precision_prior + precision_likelihood

            posterior_variance = 1.0 / precision_posterior
            posterior_mean = posterior_variance * (
                precision_prior * posterior_mean + precision_likelihood * likelihood_mean
            )

        fused_timestamp = posterior_mean
        fused_precision = math.sqrt(posterior_variance)

        # Confidence based on precision improvement
        confidence = prior_variance / posterior_variance if posterior_variance > 0 else 0.5
        confidence = min(1.0, confidence / len(signals))  # Normalize by number of signals

        return FusionResult(
            fused_timestamp=float(fused_timestamp),
            fused_precision=float(fused_precision),
            fusion_confidence=float(confidence),
            fusion_quality=FusionQuality.GOOD,
            input_signals_count=len(signals),
            fusion_algorithm=FusionAlgorithm.BAYESIAN_FUSION,
            processing_time=0.0,
            fusion_metadata={
                'prior_variance': float(prior_variance),
                'posterior_variance': float(posterior_variance),
                'precision_improvement': float(prior_variance / posterior_variance) if posterior_variance > 0 else 1.0
            }
        )

    def _adaptive_fusion(self, signals: List[TimeSignal]) -> FusionResult:
        """Adaptive fusion that selects best algorithm based on signal characteristics"""

        # Analyze signal characteristics
        precisions = [signal.precision for signal in signals]
        confidences = [signal.confidence for signal in signals]
        drift_rates = [abs(signal.drift_rate) for signal in signals]

        precision_range = max(precisions) / min(precisions) if min(precisions) > 0 else 1.0
        avg_confidence = np.mean(confidences)
        max_drift = max(drift_rates)

        # Select algorithm based on characteristics
        if max_drift > 1e-6:  # High drift rates
            selected_algorithm = FusionAlgorithm.EXTENDED_KALMAN
        elif precision_range > 100:  # Large precision differences
            selected_algorithm = FusionAlgorithm.WEIGHTED_LEAST_SQUARES
        elif avg_confidence < 0.7:  # Low confidence signals
            selected_algorithm = FusionAlgorithm.PARTICLE_FILTER
        elif len(signals) > 10:  # Many signals
            selected_algorithm = FusionAlgorithm.MAXIMUM_LIKELIHOOD
        else:  # Default case
            selected_algorithm = FusionAlgorithm.KALMAN_FILTER

        # Execute selected algorithm
        if selected_algorithm == FusionAlgorithm.KALMAN_FILTER:
            result = self._kalman_filter_fusion(signals)
        elif selected_algorithm == FusionAlgorithm.EXTENDED_KALMAN:
            result = self._extended_kalman_fusion(signals)
        elif selected_algorithm == FusionAlgorithm.WEIGHTED_LEAST_SQUARES:
            result = self._weighted_least_squares_fusion(signals)
        elif selected_algorithm == FusionAlgorithm.PARTICLE_FILTER:
            result = self._particle_filter_fusion(signals)
        elif selected_algorithm == FusionAlgorithm.MAXIMUM_LIKELIHOOD:
            result = self._maximum_likelihood_fusion(signals)
        else:
            result = self._kalman_filter_fusion(signals)

        result.fusion_algorithm = FusionAlgorithm.ADAPTIVE_FUSION
        result.fusion_metadata['selected_algorithm'] = selected_algorithm.value
        result.fusion_metadata['selection_criteria'] = {
            'precision_range': float(precision_range),
            'avg_confidence': float(avg_confidence),
            'max_drift_rate': float(max_drift)
        }

        return result

    def _assess_fusion_quality(self, precision: float) -> FusionQuality:
        """Assess fusion quality based on achieved precision"""

        for quality, threshold in self.quality_thresholds.items():
            if precision <= threshold:
                return quality

        return FusionQuality.POOR

    def _update_performance_stats(self, result: FusionResult, input_signals: List[TimeSignal]):
        """Update performance statistics"""

        self.performance_stats['total_fusions'] += 1

        if result.fusion_confidence >= 0.7:
            self.performance_stats['successful_fusions'] += 1

        # Calculate precision improvement
        best_input_precision = min(signal.precision for signal in input_signals)
        improvement_factor = result.get_precision_improvement_factor(best_input_precision)

        # Update running average
        n = self.performance_stats['total_fusions']
        prev_avg_improvement = self.performance_stats['average_precision_improvement']
        self.performance_stats['average_precision_improvement'] = (
            (prev_avg_improvement * (n - 1) + improvement_factor) / n
        )

        # Update processing time average
        prev_avg_time = self.performance_stats['average_processing_time']
        self.performance_stats['average_processing_time'] = (
            (prev_avg_time * (n - 1) + result.processing_time) / n
        )

    def fuse_buffered_signals(self,
                             signal_ids: List[str] = None,
                             algorithm: FusionAlgorithm = FusionAlgorithm.ADAPTIVE_FUSION,
                             max_age_seconds: float = 10.0) -> Optional[FusionResult]:
        """Fuse signals from buffer"""

        # Get all signal IDs if not specified
        if signal_ids is None:
            signal_ids = list(self.signal_buffer.keys())

        # Collect recent signals
        current_time = time.time()
        signals_to_fuse = []

        for signal_id in signal_ids:
            if signal_id in self.signal_buffer:
                # Get most recent signal within age limit
                buffer = self.signal_buffer[signal_id]
                for signal in reversed(buffer):  # Start from most recent
                    age = current_time - signal.measurement_time
                    if age <= max_age_seconds:
                        signals_to_fuse.append(signal)
                        break  # Only use most recent signal from each source

        if len(signals_to_fuse) >= self.minimum_signals_for_fusion:
            return self.fuse_signals(signals_to_fuse, algorithm)
        else:
            return None

    def get_real_time_fused_time(self,
                                algorithm: FusionAlgorithm = FusionAlgorithm.KALMAN_FILTER) -> Tuple[float, float]:
        """Get real-time fused timestamp and precision"""

        fusion_result = self.fuse_buffered_signals(algorithm=algorithm)

        if fusion_result:
            return fusion_result.fused_timestamp, fusion_result.fused_precision
        else:
            # Fallback to system time
            return time.time(), 1e-3  # 1ms precision fallback

    def benchmark_fusion_algorithms(self, test_signals: List[TimeSignal]) -> Dict:
        """Benchmark all fusion algorithms with test signals"""

        benchmark_results = {}

        for algorithm in FusionAlgorithm:
            try:
                start_time = time.time()
                result = self.fuse_signals(test_signals, algorithm)
                end_time = time.time()

                benchmark_results[algorithm.value] = {
                    'fused_timestamp': result.fused_timestamp,
                    'fused_precision': result.fused_precision,
                    'fusion_confidence': result.fusion_confidence,
                    'fusion_quality': result.fusion_quality.value,
                    'processing_time': end_time - start_time,
                    'precision_improvement': result.get_precision_improvement_factor(
                        min(signal.precision for signal in test_signals)
                    )
                }
            except Exception as e:
                benchmark_results[algorithm.value] = {
                    'error': str(e),
                    'processing_time': 0.0
                }

        return benchmark_results

    def get_fusion_performance_report(self) -> Dict:
        """Get comprehensive fusion performance report"""

        # Algorithm usage statistics
        algorithm_usage = {}
        for result in self.fusion_history:
            algo = result.fusion_algorithm.value
            algorithm_usage[algo] = algorithm_usage.get(algo, 0) + 1

        # Quality distribution
        quality_distribution = {}
        for result in self.fusion_history:
            quality = result.fusion_quality.value
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1

        # Precision improvements
        precision_improvements = []
        for result in self.fusion_history:
            if 'precision_improvement' in result.fusion_metadata:
                precision_improvements.append(result.fusion_metadata['precision_improvement'])

        return {
            'performance_statistics': self.performance_stats,
            'fusion_history_length': len(self.fusion_history),
            'algorithm_usage_distribution': algorithm_usage,
            'fusion_quality_distribution': quality_distribution,
            'signal_buffer_status': {
                'total_buffers': len(self.signal_buffer),
                'total_buffered_signals': sum(len(buffer) for buffer in self.signal_buffer.values())
            },
            'precision_analysis': {
                'precision_improvements': precision_improvements,
                'average_improvement': np.mean(precision_improvements) if precision_improvements else 0.0,
                'max_improvement': max(precision_improvements) if precision_improvements else 0.0,
                'min_improvement': min(precision_improvements) if precision_improvements else 0.0
            },
            'kalman_states_active': len(self.kalman_states),
            'system_configuration': {
                'buffer_size': self.buffer_size,
                'minimum_signals_for_fusion': self.minimum_signals_for_fusion,
                'maximum_fusion_latency': self.maximum_fusion_latency
            }
        }


def create_signal_fusion_system() -> SignalFusionEngine:
    """Create signal fusion system for multi-precision time signal combination"""
    return SignalFusionEngine()
