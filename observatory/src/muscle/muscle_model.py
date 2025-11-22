"""
Multi-Scale Oscillatory Coupling Framework for Muscle Modeling

Extends classical Hill-type muscle models with oscillatory coupling theory across
10 hierarchical scales to capture the emergent dynamics of muscle force generation
and coordination.

Author: Based on oscillatory framework described in README.md
Extends: Thelen2003 and McLean2003 muscle models from notebooks
"""

import numpy as np
from scipy import signal, fft
from scipy.integrate import ode, odeint
from scipy.interpolate import interp1d
import warnings
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class OscillatoryScale:
    """Represents one scale in the oscillatory hierarchy."""
    name: str
    freq_min: float  # Hz
    freq_max: float  # Hz
    scale_index: int


class OscillatoryHierarchy:
    """
    10-Scale Oscillatory Hierarchy for biomechanical systems.

    Maps the oscillatory scales from quantum membrane to allometric,
    as defined in the theoretical framework.
    """

    SCALES = [
        OscillatoryScale("Quantum Membrane", 1e12, 1e15, 0),
        OscillatoryScale("Intracellular", 1e3, 1e6, 1),
        OscillatoryScale("Cellular", 1e-1, 1e2, 2),
        OscillatoryScale("Tissue", 1e-2, 1e1, 3),
        OscillatoryScale("Neural", 1, 100, 4),
        OscillatoryScale("Neuromuscular", 0.01, 20, 5),
        OscillatoryScale("Cardiovascular", 0.01, 5, 6),
        OscillatoryScale("Locomotor", 0.5, 3, 7),
        OscillatoryScale("Circadian", 1e-5, 1e-5, 8),
        OscillatoryScale("Allometric", 1e-8, 1e-5, 9)
    ]

    @classmethod
    def get_scale(cls, index: int) -> OscillatoryScale:
        """Get scale by index."""
        return cls.SCALES[index]

    @classmethod
    def get_relevant_muscle_scales(cls) -> List[OscillatoryScale]:
        """Get scales most relevant for muscle dynamics."""
        # Tissue (3), Neural (4), Neuromuscular (5), Cardiovascular (6), Locomotor (7)
        return [cls.SCALES[i] for i in range(3, 8)]


class OscillatoryCouplingAnalyzer:
    """
    Analyzes coupling strength between oscillatory scales.

    Implements coupling strength computation:
    C_ij(t) = |1/T ∫₀ᵀ A_i(φ_j(t+τ)) e^(iφ_i(t+τ)) dτ|
    """

    def __init__(self, sampling_rate: float = 1000.0):
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate

    def extract_phase(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Extract instantaneous phase from signal using Hilbert transform.

        Parameters
        ----------
        signal_data : array
            Time series data

        Returns
        -------
        phase : array
            Instantaneous phase
        """
        analytic_signal = signal.hilbert(signal_data)
        return np.unwrap(np.angle(analytic_signal))

    def bandpass_filter(self, data: np.ndarray, freq_min: float,
                       freq_max: float) -> np.ndarray:
        """
        Bandpass filter signal for specific frequency range.

        Parameters
        ----------
        data : array
            Input signal
        freq_min, freq_max : float
            Frequency band limits in Hz

        Returns
        -------
        filtered : array
            Bandpass filtered signal
        """
        nyquist = self.sampling_rate / 2
        low = freq_min / nyquist
        high = freq_max / nyquist

        # Ensure valid frequency range
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, low + 0.001, 0.999)

        try:
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            filtered = signal.sosfiltfilt(sos, data)
        except Exception as e:
            warnings.warn(f"Filtering failed: {e}. Returning original signal.")
            filtered = data

        return filtered

    def compute_coupling_strength(self, signal_i: np.ndarray,
                                 signal_j: np.ndarray,
                                 window_size: Optional[int] = None) -> float:
        """
        Compute coupling strength between two oscillatory signals.

        Parameters
        ----------
        signal_i, signal_j : array
            Two signals to analyze coupling
        window_size : int, optional
            Integration window size (samples)

        Returns
        -------
        coupling : float
            Coupling strength C_ij
        """
        if window_size is None:
            window_size = len(signal_i)

        # Extract phases
        phi_i = self.extract_phase(signal_i)
        phi_j = self.extract_phase(signal_j)

        # Compute amplitude of signal i
        analytic_i = signal.hilbert(signal_i)
        A_i = np.abs(analytic_i)

        # Compute coupling integrand
        # A_i(φ_j) * e^(i*φ_i)
        integrand = A_i * np.exp(1j * phi_i)

        # Integrate over window
        coupling = np.abs(np.mean(integrand))

        # Normalize by mean amplitude
        coupling = coupling / (np.mean(A_i) + 1e-10)

        return coupling

    def compute_coupling_matrix(self, signals: Dict[str, np.ndarray],
                               scales: List[OscillatoryScale]) -> np.ndarray:
        """
        Compute full coupling matrix between all scales.

        Parameters
        ----------
        signals : dict
            Dictionary mapping scale names to signal data
        scales : list
            List of OscillatoryScale objects

        Returns
        -------
        coupling_matrix : array (n_scales x n_scales)
            Coupling strength matrix
        """
        n_scales = len(scales)
        coupling_matrix = np.zeros((n_scales, n_scales))

        for i, scale_i in enumerate(scales):
            for j, scale_j in enumerate(scales):
                if scale_i.name in signals and scale_j.name in signals:
                    coupling_matrix[i, j] = self.compute_coupling_strength(
                        signals[scale_i.name],
                        signals[scale_j.name]
                    )

        return coupling_matrix


class GearRatioTransform:
    """
    Implements gear ratio transformations for O(1) complexity navigation
    between oscillatory scales.

    R_{i→j} = ω_i / ω_j
    """

    @staticmethod
    def compute_gear_ratio(freq_i: float, freq_j: float) -> float:
        """
        Compute gear ratio between two frequencies.

        Parameters
        ----------
        freq_i, freq_j : float
            Frequencies of two oscillatory scales

        Returns
        -------
        ratio : float
            Gear ratio R_{i→j}
        """
        return freq_i / (freq_j + 1e-10)

    @staticmethod
    def compute_scale_gear_ratios(scales: List[OscillatoryScale]) -> np.ndarray:
        """
        Compute gear ratios between adjacent scales.

        Parameters
        ----------
        scales : list
            Ordered list of oscillatory scales

        Returns
        -------
        gear_ratios : array
            Array of gear ratios between adjacent scales
        """
        n_scales = len(scales)
        gear_ratios = np.zeros(n_scales - 1)

        for i in range(n_scales - 1):
            # Use geometric mean of frequency range
            freq_i = np.sqrt(scales[i].freq_min * scales[i].freq_max)
            freq_j = np.sqrt(scales[i+1].freq_min * scales[i+1].freq_max)
            gear_ratios[i] = GearRatioTransform.compute_gear_ratio(freq_i, freq_j)

        return gear_ratios


class StateSpaceCoordinates:
    """
    Tri-dimensional state space for muscle dynamics.

    s = (s_knowledge, s_time, s_entropy)
    """

    @staticmethod
    def compute_knowledge_dimension(coupling_matrix: np.ndarray,
                                   history_length: int = 10) -> float:
        """
        Compute knowledge dimension based on coupling structure.

        Higher coupling = more information about system state.

        Parameters
        ----------
        coupling_matrix : array
            Matrix of coupling strengths
        history_length : int
            Number of historical time steps to consider

        Returns
        -------
        s_knowledge : float
            Knowledge coordinate
        """
        # Use spectral properties of coupling matrix
        eigenvalues = np.linalg.eigvalsh(coupling_matrix)

        # Participation ratio as measure of effective dimensionality
        lambda_sum = np.sum(eigenvalues)
        lambda_sq_sum = np.sum(eigenvalues**2)

        s_knowledge = lambda_sum**2 / (lambda_sq_sum + 1e-10)

        return s_knowledge

    @staticmethod
    def compute_time_dimension(signal_data: np.ndarray,
                              dt: float) -> float:
        """
        Compute time dimension from signal autocorrelation.

        Parameters
        ----------
        signal_data : array
            Time series data
        dt : float
            Time step

        Returns
        -------
        s_time : float
            Time coordinate (characteristic timescale)
        """
        # Autocorrelation
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        # Find first zero crossing or 1/e decay
        threshold = np.exp(-1)
        indices = np.where(autocorr < threshold)[0]

        if len(indices) > 0:
            s_time = indices[0] * dt
        else:
            s_time = len(autocorr) * dt

        return s_time

    @staticmethod
    def compute_entropy_dimension(signal_data: np.ndarray) -> float:
        """
        Compute entropy dimension using sample entropy.

        Parameters
        ----------
        signal_data : array
            Time series data

        Returns
        -------
        s_entropy : float
            Entropy coordinate
        """
        # Simplified sample entropy
        # Normalize signal
        signal_norm = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-10)

        # Compute approximate entropy
        m = 2  # Pattern length
        r = 0.2 * np.std(signal_norm)  # Tolerance

        def _maxdist(x_i, x_j):
            return np.max(np.abs(x_i - x_j))

        N = len(signal_norm)

        # Pattern matching for m
        phi_m = []
        for m_val in [m, m + 1]:
            patterns = np.array([signal_norm[i:i+m_val] for i in range(N - m_val)])
            C = np.zeros(len(patterns))

            for i in range(len(patterns)):
                matches = 0
                for j in range(len(patterns)):
                    if _maxdist(patterns[i], patterns[j]) <= r:
                        matches += 1
                C[i] = matches / len(patterns)

            phi = np.mean(np.log(C + 1e-10))
            phi_m.append(phi)

        s_entropy = phi_m[0] - phi_m[1]

        return s_entropy

    @staticmethod
    def compute_state_coordinates(signal_data: np.ndarray,
                                 coupling_matrix: np.ndarray,
                                 dt: float) -> Tuple[float, float, float]:
        """
        Compute full state space coordinates.

        Parameters
        ----------
        signal_data : array
            Time series data
        coupling_matrix : array
            Coupling matrix between scales
        dt : float
            Time step

        Returns
        -------
        coordinates : tuple (s_knowledge, s_time, s_entropy)
            State space coordinates
        """
        s_knowledge = StateSpaceCoordinates.compute_knowledge_dimension(coupling_matrix)
        s_time = StateSpaceCoordinates.compute_time_dimension(signal_data, dt)
        s_entropy = StateSpaceCoordinates.compute_entropy_dimension(signal_data)

        return (s_knowledge, s_time, s_entropy)


class OscillatoryMuscleModel:
    """
    Extended Hill-type muscle model with multi-scale oscillatory coupling.

    Integrates classical Hill muscle mechanics with oscillatory dynamics across
    hierarchical scales (tissue, neural, neuromuscular, cardiovascular, locomotor).

    Key extensions:
    1. Activation dynamics modulated by neural oscillations
    2. Force generation coupled to tissue-scale oscillations
    3. Length-velocity dynamics influenced by neuromuscular coupling
    4. Cardiovascular coupling affects fatigue and recovery
    5. Locomotor rhythm entrains overall muscle behavior
    """

    def __init__(self, parameters: Optional[Dict] = None):
        """
        Initialize oscillatory muscle model.

        Parameters
        ----------
        parameters : dict, optional
            Muscle model parameters (follows Thelen2003 convention)
        """
        # Default Thelen2003 parameters
        self.P = {
            'fm0': 7400.0,      # Maximum isometric force (N)
            'lmopt': 0.093,     # Optimal fiber length (m)
            'ltslack': 0.223,   # Tendon slack length (m)
            'alpha0': 0.0,      # Pennation angle (rad)
            'vmmax': 10.0,      # Max contraction velocity (lmopt/s)
            'af': 0.25,         # Force-velocity shape factor
            'fmlen': 1.4,       # Max eccentric force multiplier
            'gammal': 0.45,     # Force-length shape factor
            'kpe': 5.0,         # Parallel element stiffness
            'epsm0': 0.6,       # Passive muscle strain
            'epst0': 0.04,      # Tendon strain at fm0
            'kttoe': 3.0,       # Tendon toe region shape
            't_act': 0.015,     # Activation time constant (s)
            't_deact': 0.050,   # Deactivation time constant (s)
            'u_max': 1.0,       # Max excitation
            'u_min': 0.01,      # Min excitation
        }

        if parameters:
            self.P.update(parameters)

        # Initialize oscillatory components
        self.coupling_analyzer = OscillatoryCouplingAnalyzer()
        self.gear_transform = GearRatioTransform()

        # Get relevant scales for muscle
        self.scales = OscillatoryHierarchy.get_relevant_muscle_scales()

        # State variables
        self.coupling_matrix = None
        self.state_coordinates = None
        self.oscillatory_signals = {}

        # History for temporal analysis
        self.time_history = []
        self.force_history = []
        self.activation_history = []
        self.length_history = []

    def force_length_ce(self, lm_norm: float) -> float:
        """
        Contractile element force-length relationship.

        f_l = exp(-(lm_norm - 1)^2 / γ)

        Parameters
        ----------
        lm_norm : float
            Normalized muscle fiber length (lm / lmopt)

        Returns
        -------
        fl : float
            Normalized force
        """
        return np.exp(-(lm_norm - 1)**2 / self.P['gammal'])

    def force_length_pe(self, lm_norm: float) -> float:
        """
        Parallel element (passive) force-length relationship.

        Parameters
        ----------
        lm_norm : float
            Normalized muscle fiber length

        Returns
        -------
        fpe : float
            Normalized passive force
        """
        if lm_norm <= 1:
            return 0.0
        else:
            kpe = self.P['kpe']
            epsm0 = self.P['epsm0']
            return (np.exp(kpe * (lm_norm - 1) / epsm0) - 1) / (np.exp(kpe) - 1)

    def force_length_se(self, lt: float) -> float:
        """
        Series element (tendon) force-length relationship.

        Parameters
        ----------
        lt : float
            Tendon length

        Returns
        -------
        fse : float
            Normalized tendon force
        """
        ltslack = self.P['ltslack']
        epst0 = self.P['epst0']
        kttoe = self.P['kttoe']

        epst = (lt - ltslack) / ltslack
        fttoe = 0.33

        # OpenSim Thelen2003Muscle values
        epsttoe = 0.99 * epst0 * np.e**3 / (1.66 * np.e**3 - 0.67)
        ktlin = 0.67 / (epst0 - epsttoe)

        if epst <= 0:
            return 0.0
        elif epst <= epsttoe:
            return fttoe / (np.exp(kttoe) - 1) * (np.exp(kttoe * epst / epsttoe) - 1)
        else:
            return ktlin * (epst - epsttoe) + fttoe

    def force_velocity_ce(self, vm: float, fl: float, a: float = 1.0) -> float:
        """
        Contractile element force-velocity relationship.

        Parameters
        ----------
        vm : float
            Muscle fiber velocity
        fl : float
            Force from length relationship
        a : float
            Activation level

        Returns
        -------
        fv : float
            Normalized force from velocity
        """
        lmopt = self.P['lmopt']
        vmmax = self.P['vmmax'] * lmopt
        af = self.P['af']
        fmlen = self.P['fmlen']

        if vm <= 0:  # Concentric
            fv = af * a * fl * (4 * vm + vmmax * (3 * a + 1)) / \
                 (-4 * vm + vmmax * af * (3 * a + 1))
        else:  # Eccentric
            fv = a * fl * (af * vmmax * (3 * a * fmlen - 3 * a + fmlen - 1) +
                          8 * vm * fmlen * (af + 1)) / \
                 (af * vmmax * (3 * a * fmlen - 3 * a + fmlen - 1) +
                  8 * vm * (af + 1))

        return fv

    def velocity_from_force(self, fm_norm: float, fl: float, a: float = 1.0) -> float:
        """
        Invert force-velocity to get velocity from force.

        Parameters
        ----------
        fm_norm : float
            Normalized muscle force
        fl : float
            Force-length factor
        a : float
            Activation level

        Returns
        -------
        vm : float
            Muscle fiber velocity
        """
        lmopt = self.P['lmopt']
        vmmax = self.P['vmmax'] * lmopt
        af = self.P['af']
        fmlen = self.P['fmlen']

        if fm_norm <= a * fl:  # Concentric/isometric
            if fm_norm > 0:
                b = a * fl + fm_norm / af
            else:
                b = a * fl
        else:  # Eccentric
            asyE_thresh = 0.95
            if fm_norm < a * fl * fmlen * asyE_thresh:
                b = (2 + 2/af) * (a * fl * fmlen - fm_norm) / (fmlen - 1)
            else:
                fm0 = a * fl * fmlen * asyE_thresh
                b = (2 + 2/af) * (a * fl * fmlen - fm0) / (fmlen - 1)

        vm = (0.25 + 0.75 * a) * (fm_norm - a * fl) / b * vmmax

        return vm

    def activation_dynamics_oscillatory(self, t: float, a: float,
                                       u: float,
                                       neural_osc: Optional[float] = None) -> float:
        """
        Activation dynamics with neural oscillation modulation.

        da/dt = (u - a) / τ(a, u) * (1 + β * sin(ω_neural * t))

        Neural oscillations (1-100 Hz) modulate the activation/deactivation rates.

        Parameters
        ----------
        t : float
            Time
        a : float
            Current activation
        u : float
            Excitation signal
        neural_osc : float, optional
            Neural oscillation amplitude

        Returns
        -------
        adot : float
            Activation rate of change
        """
        t_act = self.P['t_act']
        t_deact = self.P['t_deact']

        # Base activation dynamics (Thelen2003)
        if u > a:
            tau = t_act * (0.5 + 1.5 * a)
        else:
            tau = t_deact / (0.5 + 1.5 * a)

        adot_base = (u - a) / tau

        # Neural oscillation modulation
        if neural_osc is not None:
            # Neural oscillations modulate the time constant
            # β controls coupling strength between neural and neuromuscular scales
            beta = 0.1  # Coupling strength parameter
            neural_modulation = 1.0 + beta * neural_osc
            adot = adot_base * neural_modulation
        else:
            adot = adot_base

        return adot

    def compute_decoupling_threshold(self, frequencies: np.ndarray) -> float:
        """
        Compute critical coupling threshold for decoupling detection.

        C_critical = 1/(N-1) * √(Σω_k² / Σω_k)

        Parameters
        ----------
        frequencies : array
            Dominant frequencies at each scale

        Returns
        -------
        c_critical : float
            Critical coupling threshold
        """
        N = len(frequencies)
        omega_sum = np.sum(frequencies)
        omega_sq_sum = np.sum(frequencies**2)

        c_critical = (1.0 / (N - 1)) * np.sqrt(omega_sq_sum / (omega_sum + 1e-10))

        return c_critical

    def extract_oscillatory_components(self, force_history: np.ndarray,
                                      time_history: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose force signal into oscillatory components at different scales.

        Parameters
        ----------
        force_history : array
            Time series of muscle force
        time_history : array
            Corresponding time points

        Returns
        -------
        signals : dict
            Signals at each relevant oscillatory scale
        """
        signals = {}

        # Compute sampling rate
        dt = np.mean(np.diff(time_history))
        fs = 1.0 / dt
        self.coupling_analyzer.sampling_rate = fs

        # Extract signals at each relevant scale
        for scale in self.scales:
            # Bandpass filter for this scale
            filtered = self.coupling_analyzer.bandpass_filter(
                force_history,
                scale.freq_min,
                scale.freq_max
            )
            signals[scale.name] = filtered

        return signals

    def update_oscillatory_state(self, t: float, force: float,
                                activation: float, length: float):
        """
        Update oscillatory analysis with new state.

        Parameters
        ----------
        t : float
            Current time
        force : float
            Current muscle force
        activation : float
            Current activation
        length : float
            Current muscle length
        """
        self.time_history.append(t)
        self.force_history.append(force)
        self.activation_history.append(activation)
        self.length_history.append(length)

        # Update coupling analysis periodically (every ~100 samples)
        if len(self.time_history) > 100 and len(self.time_history) % 100 == 0:
            self._update_coupling_analysis()

    def _update_coupling_analysis(self):
        """Internal: Update coupling matrix and state coordinates."""
        force_array = np.array(self.force_history)
        time_array = np.array(self.time_history)

        # Extract oscillatory components
        self.oscillatory_signals = self.extract_oscillatory_components(
            force_array, time_array
        )

        # Compute coupling matrix
        self.coupling_matrix = self.coupling_analyzer.compute_coupling_matrix(
            self.oscillatory_signals, self.scales
        )

        # Compute state space coordinates
        dt = np.mean(np.diff(time_array))
        self.state_coordinates = StateSpaceCoordinates.compute_state_coordinates(
            force_array, self.coupling_matrix, dt
        )

    def get_coupling_modulation(self, scale_name: str) -> float:
        """
        Get current coupling strength modulation factor for a scale.

        Parameters
        ----------
        scale_name : str
            Name of the oscillatory scale

        Returns
        -------
        modulation : float
            Coupling-based modulation factor (around 1.0)
        """
        if self.coupling_matrix is None:
            return 1.0

        # Find scale index
        scale_idx = None
        for i, scale in enumerate(self.scales):
            if scale.name == scale_name:
                scale_idx = i
                break

        if scale_idx is None:
            return 1.0

        # Average coupling to other scales
        avg_coupling = np.mean(self.coupling_matrix[scale_idx, :])

        # Modulation factor (stronger coupling = stronger modulation)
        # Range: [0.8, 1.2]
        modulation = 0.8 + 0.4 * avg_coupling

        return modulation

    def simulate_muscle_with_coupling(self,
                                     excitation_func,
                                     lmt_func,
                                     t_span: Tuple[float, float] = (0, 3.0),
                                     lm0: Optional[float] = None,
                                     dt: float = 0.001,
                                     enable_coupling: bool = True) -> Dict:
        """
        Simulate muscle dynamics with oscillatory coupling.

        Parameters
        ----------
        excitation_func : callable
            Function u(t) returning excitation level
        lmt_func : callable
            Function lmt(t) returning muscle-tendon length
        t_span : tuple
            (t_start, t_end) simulation time span
        lm0 : float, optional
            Initial muscle fiber length
        dt : float
            Time step
        enable_coupling : bool
            Whether to enable oscillatory coupling modulation

        Returns
        -------
        results : dict
            Simulation results including force, length, velocity, activation, coupling
        """
        # Initialize
        if lm0 is None:
            lm0 = self.P['lmopt']

        lmopt = self.P['lmopt']
        ltslack = self.P['ltslack']
        fm0 = self.P['fm0']
        alpha0 = self.P['alpha0']

        # Time array
        t_array = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t_array)

        # Storage arrays
        lm_array = np.zeros(n_steps)
        lt_array = np.zeros(n_steps)
        vm_array = np.zeros(n_steps)
        fm_array = np.zeros(n_steps)
        a_array = np.zeros(n_steps)
        lmt_array = np.zeros(n_steps)

        # Coupling analysis storage
        coupling_strength_array = np.zeros(n_steps)
        state_coords_array = np.zeros((n_steps, 3))

        # Initial conditions
        lm_array[0] = lm0
        a_array[0] = self.P['u_min']
        lmt_array[0] = lmt_func(t_array[0])
        lt_array[0] = lmt_array[0] - lm_array[0] * np.cos(alpha0)

        # Integration using Euler method for clarity
        # (could use more sophisticated ODE solver)
        for i in range(1, n_steps):
            t = t_array[i]

            # Current state
            lm = lm_array[i-1]
            a = a_array[i-1]

            # Muscle-tendon length
            lmt = lmt_func(t)
            lmt_array[i] = lmt

            # Excitation
            u = excitation_func(t)

            # Activation dynamics with neural oscillation
            if enable_coupling and len(self.time_history) > 10:
                # Sample neural oscillation (simplified: use filtered activation history)
                neural_signal = np.array(self.activation_history[-min(100, len(self.activation_history)):])
                if len(neural_signal) > 10:
                    neural_freq = OscillatoryHierarchy.SCALES[4].freq_min  # Neural scale
                    neural_osc = np.sin(2 * np.pi * neural_freq * t) * np.std(neural_signal)
                else:
                    neural_osc = 0.0
            else:
                neural_osc = 0.0

            adot = self.activation_dynamics_oscillatory(t, a, u, neural_osc)
            a_new = a + adot * dt
            a_new = np.clip(a_new, self.P['u_min'], self.P['u_max'])
            a_array[i] = a_new

            # Constrain muscle length
            if lm < 0.1 * lmopt:
                lm = 0.1 * lmopt

            # Pennation angle (simplified: constant for now)
            alpha = alpha0

            # Tendon length
            lt = lmt - lm * np.cos(alpha)
            lt_array[i] = lt

            # Forces
            fl = self.force_length_ce(lm / lmopt)
            fpe = self.force_length_pe(lm / lmopt)
            fse = self.force_length_se(lt)

            # Muscle force must equal tendon force at equilibrium
            fce_norm = fse / np.cos(alpha) - fpe

            # Coupling modulation
            if enable_coupling and self.coupling_matrix is not None:
                tissue_modulation = self.get_coupling_modulation("Tissue")
                neuromuscular_modulation = self.get_coupling_modulation("Neuromuscular")
                fce_norm *= tissue_modulation

            # Velocity from force-velocity relation
            vm = self.velocity_from_force(fce_norm, fl, a_new)
            vm_array[i] = vm

            # Update muscle length
            lm_new = lm + vm * dt
            lm_array[i] = lm_new

            # Total muscle force
            fm = (fce_norm + fpe) * fm0
            if enable_coupling and self.coupling_matrix is not None:
                fm *= neuromuscular_modulation
            fm_array[i] = fm

            # Update oscillatory state
            self.update_oscillatory_state(t, fm, a_new, lm_new)

            # Store coupling metrics
            if self.coupling_matrix is not None:
                coupling_strength_array[i] = np.mean(self.coupling_matrix)
            if self.state_coordinates is not None:
                state_coords_array[i] = self.state_coordinates

        # Prepare results
        results = {
            'time': t_array,
            'muscle_length': lm_array,
            'tendon_length': lt_array,
            'muscle_tendon_length': lmt_array,
            'muscle_velocity': vm_array,
            'muscle_force': fm_array,
            'activation': a_array,
            'coupling_strength': coupling_strength_array,
            'state_coordinates': state_coords_array,
            'oscillatory_signals': self.oscillatory_signals,
            'coupling_matrix': self.coupling_matrix,
            'scales': [s.name for s in self.scales]
        }

        return results

    def compute_performance_metrics(self, results: Dict) -> Dict:
        """
        Compute performance metrics from simulation results.

        Parameters
        ----------
        results : dict
            Simulation results from simulate_muscle_with_coupling

        Returns
        -------
        metrics : dict
            Performance metrics including power, work, coupling efficiency
        """
        force = results['muscle_force']
        velocity = results['muscle_velocity']
        time = results['time']
        coupling = results['coupling_strength']

        # Power = Force × Velocity
        power = force * velocity

        # Work = ∫ Power dt
        work = np.trapz(power, time)

        # Peak force and power
        peak_force = np.max(force)
        peak_power = np.max(np.abs(power))

        # Coupling efficiency: higher coupling = more efficient coordination
        if len(coupling[coupling > 0]) > 0:
            avg_coupling = np.mean(coupling[coupling > 0])
        else:
            avg_coupling = 0.0

        # State space volume (measure of system complexity)
        state_coords = results['state_coordinates']
        if np.any(state_coords):
            # Volume of state space occupied
            state_volume = np.prod(np.ptp(state_coords, axis=0))
        else:
            state_volume = 0.0

        metrics = {
            'peak_force': peak_force,
            'peak_power': peak_power,
            'total_work': work,
            'average_coupling': avg_coupling,
            'state_space_volume': state_volume,
            'coupling_efficiency': avg_coupling / (state_volume + 1.0)  # Normalized
        }

        return metrics


def example_simulation():
    """
    Example: Simulate muscle with oscillatory coupling during isometric contraction.
    """
    import matplotlib.pyplot as plt

    # Create muscle model
    muscle = OscillatoryMuscleModel()

    # Define excitation (step input)
    def excitation(t):
        if 0.5 <= t <= 2.0:
            return 1.0
        else:
            return 0.01

    # Define muscle-tendon length (isometric)
    lmt0 = 0.31
    def lmt(t):
        return lmt0

    print("Running simulation with oscillatory coupling...")
    results_coupled = muscle.simulate_muscle_with_coupling(
        excitation, lmt,
        t_span=(0, 3.0),
        enable_coupling=True
    )

    print("Running simulation without oscillatory coupling...")
    muscle2 = OscillatoryMuscleModel()
    results_uncoupled = muscle2.simulate_muscle_with_coupling(
        excitation, lmt,
        t_span=(0, 3.0),
        enable_coupling=False
    )

    # Compute metrics
    metrics_coupled = muscle.compute_performance_metrics(results_coupled)
    metrics_uncoupled = muscle2.compute_performance_metrics(results_uncoupled)

    print("\n=== Performance Metrics ===")
    print("\nWith Oscillatory Coupling:")
    for key, value in metrics_coupled.items():
        print(f"  {key}: {value:.4f}")

    print("\nWithout Oscillatory Coupling:")
    for key, value in metrics_uncoupled.items():
        print(f"  {key}: {value:.4f}")

    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    t_c = results_coupled['time']
    t_u = results_uncoupled['time']

    # Force comparison
    axes[0, 0].plot(t_c, results_coupled['muscle_force'], label='With Coupling')
    axes[0, 0].plot(t_u, results_uncoupled['muscle_force'], '--', label='Without Coupling')
    axes[0, 0].set_ylabel('Force (N)')
    axes[0, 0].legend()
    axes[0, 0].set_title('Muscle Force')
    axes[0, 0].grid(True)

    # Activation
    axes[0, 1].plot(t_c, results_coupled['activation'], label='With Coupling')
    axes[0, 1].plot(t_u, results_uncoupled['activation'], '--', label='Without Coupling')
    axes[0, 1].set_ylabel('Activation')
    axes[0, 1].legend()
    axes[0, 1].set_title('Muscle Activation')
    axes[0, 1].grid(True)

    # Muscle length
    axes[1, 0].plot(t_c, results_coupled['muscle_length'])
    axes[1, 0].set_ylabel('Length (m)')
    axes[1, 0].set_title('Muscle Fiber Length')
    axes[1, 0].grid(True)

    # Coupling strength
    axes[1, 1].plot(t_c, results_coupled['coupling_strength'])
    axes[1, 1].set_ylabel('Coupling Strength')
    axes[1, 1].set_title('Average Coupling Strength')
    axes[1, 1].grid(True)

    # State space coordinates
    state_coords = results_coupled['state_coordinates']
    if np.any(state_coords):
        axes[2, 0].plot(t_c, state_coords[:, 0], label='Knowledge')
        axes[2, 0].plot(t_c, state_coords[:, 1], label='Time')
        axes[2, 0].plot(t_c, state_coords[:, 2], label='Entropy')
        axes[2, 0].set_ylabel('State Coordinate')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].legend()
        axes[2, 0].set_title('State Space Coordinates')
        axes[2, 0].grid(True)

    # Coupling matrix (final)
    if results_coupled['coupling_matrix'] is not None:
        im = axes[2, 1].imshow(results_coupled['coupling_matrix'], cmap='hot', aspect='auto')
        axes[2, 1].set_title('Coupling Matrix')
        axes[2, 1].set_xlabel('Scale Index')
        axes[2, 1].set_ylabel('Scale Index')
        plt.colorbar(im, ax=axes[2, 1])
        # Add scale names
        scale_names = [s[:3] for s in results_coupled['scales']]  # Abbreviated
        axes[2, 1].set_xticks(range(len(scale_names)))
        axes[2, 1].set_yticks(range(len(scale_names)))
        axes[2, 1].set_xticklabels(scale_names, rotation=45)
        axes[2, 1].set_yticklabels(scale_names)

    plt.tight_layout()
    plt.savefig('oscillatory_muscle_simulation.png', dpi=150)
    plt.show()

    print("\nPlot saved as 'oscillatory_muscle_simulation.png'")


if __name__ == "__main__":
    # Run example simulation
    example_simulation()
