"""
Body Segmentation with Oscillatory Coupling Framework

Extends traditional body segment parameter models with multi-scale oscillatory
coupling to capture coordination dynamics across body segments during movement.

Integrates with muscle_model.py to provide complete musculoskeletal system
with oscillatory coupling across hierarchical scales.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class BodySegment:
    """
    Represents a body segment with its physical and oscillatory properties.
    """
    name: str
    mass: float  # kg
    length: float  # m
    com_ratio: float  # Center of mass position as ratio of length from proximal end
    radius_gyration_ratio: float  # Radius of gyration as ratio of length
    
    # Oscillatory properties
    natural_frequency: float  # Hz - natural oscillation frequency of segment
    damping_ratio: float  # Damping coefficient
    
    def moment_of_inertia(self) -> float:
        """Compute moment of inertia about center of mass."""
        return self.mass * (self.radius_gyration_ratio * self.length)**2
    
    def com_position(self) -> float:
        """Compute center of mass position from proximal end."""
        return self.com_ratio * self.length


class BodySegmentParameters:
    """
    Standard body segment parameters based on anthropometric data.
    
    Based on de Leva (1996) adjustments to Zatsiorsky parameters.
    """
    
    # Segment parameters as ratios of body mass and segment length
    SEGMENT_DATA = {
        'head': {
            'mass_ratio': 0.0694,
            'com_ratio': 0.5002,
            'rgyration_ratio': 0.303,
            'natural_freq': 4.0,  # Head-neck oscillation ~4 Hz
            'damping': 0.3
        },
        'trunk': {
            'mass_ratio': 0.4346,
            'com_ratio': 0.4486,
            'rgyration_ratio': 0.372,
            'natural_freq': 2.5,  # Trunk oscillation ~2-3 Hz
            'damping': 0.4
        },
        'upper_arm': {
            'mass_ratio': 0.0271,
            'com_ratio': 0.5772,
            'rgyration_ratio': 0.322,
            'natural_freq': 5.0,
            'damping': 0.25
        },
        'forearm': {
            'mass_ratio': 0.0162,
            'com_ratio': 0.4574,
            'rgyration_ratio': 0.303,
            'natural_freq': 6.0,
            'damping': 0.2
        },
        'hand': {
            'mass_ratio': 0.0061,
            'com_ratio': 0.7900,
            'rgyration_ratio': 0.587,
            'natural_freq': 8.0,
            'damping': 0.15
        },
        'thigh': {
            'mass_ratio': 0.1416,
            'com_ratio': 0.4095,
            'rgyration_ratio': 0.329,
            'natural_freq': 3.5,
            'damping': 0.35
        },
        'shank': {
            'mass_ratio': 0.0433,
            'com_ratio': 0.4459,
            'rgyration_ratio': 0.255,
            'natural_freq': 5.5,
            'damping': 0.25
        },
        'foot': {
            'mass_ratio': 0.0137,
            'com_ratio': 0.4415,
            'rgyration_ratio': 0.257,
            'natural_freq': 7.0,
            'damping': 0.2
        }
    }
    
    @classmethod
    def create_segment(cls, segment_name: str, body_mass: float, 
                      segment_length: float) -> BodySegment:
        """
        Create a body segment with given parameters.
        
        Parameters
        ----------
        segment_name : str
            Name of segment (e.g., 'thigh', 'shank')
        body_mass : float
            Total body mass (kg)
        segment_length : float
            Length of this segment (m)
            
        Returns
        -------
        segment : BodySegment
            Initialized body segment
        """
        if segment_name not in cls.SEGMENT_DATA:
            raise ValueError(f"Unknown segment: {segment_name}")
        
        data = cls.SEGMENT_DATA[segment_name]
        
        return BodySegment(
            name=segment_name,
            mass=body_mass * data['mass_ratio'],
            length=segment_length,
            com_ratio=data['com_ratio'],
            radius_gyration_ratio=data['rgyration_ratio'],
            natural_frequency=data['natural_freq'],
            damping_ratio=data['damping']
        )


class OscillatoryKinematicChain:
    """
    Kinematic chain of body segments with oscillatory coupling.
    
    Models coordination between segments as coupled oscillators with
    hierarchical coupling across locomotor, neuromuscular, and tissue scales.
    """
    
    def __init__(self, segments: List[BodySegment]):
        """
        Initialize kinematic chain.
        
        Parameters
        ----------
        segments : list of BodySegment
            Ordered list of segments from proximal to distal
        """
        self.segments = segments
        self.n_segments = len(segments)
        
        # Coupling matrix between segments
        self.segment_coupling = np.zeros((self.n_segments, self.n_segments))
        self._compute_segment_coupling()
        
    def _compute_segment_coupling(self):
        """
        Compute coupling strength between segments based on proximity and
        frequency relationships.
        
        Adjacent segments have stronger coupling.
        Similar frequencies enhance coupling (resonance).
        """
        for i in range(self.n_segments):
            for j in range(self.n_segments):
                if i == j:
                    self.segment_coupling[i, j] = 1.0
                else:
                    # Distance coupling (adjacent = stronger)
                    dist = abs(i - j)
                    dist_coupling = np.exp(-dist / 2.0)
                    
                    # Frequency coupling (resonance)
                    freq_i = self.segments[i].natural_frequency
                    freq_j = self.segments[j].natural_frequency
                    freq_ratio = min(freq_i, freq_j) / max(freq_i, freq_j)
                    freq_coupling = freq_ratio
                    
                    # Combined coupling
                    self.segment_coupling[i, j] = 0.7 * dist_coupling + 0.3 * freq_coupling
    
    def compute_total_mass(self) -> float:
        """Compute total mass of the chain."""
        return sum(seg.mass for seg in self.segments)
    
    def compute_total_com(self, segment_positions: np.ndarray) -> np.ndarray:
        """
        Compute center of mass of entire chain.
        
        Parameters
        ----------
        segment_positions : array (n_segments, 3)
            3D positions of each segment COM
            
        Returns
        -------
        com : array (3,)
            3D position of total COM
        """
        total_mass = self.compute_total_mass()
        weighted_pos = np.sum(
            segment_positions * np.array([seg.mass for seg in self.segments])[:, None],
            axis=0
        )
        return weighted_pos / total_mass
    
    def compute_total_momentum(self, segment_velocities: np.ndarray) -> np.ndarray:
        """
        Compute total linear momentum.
        
        Parameters
        ----------
        segment_velocities : array (n_segments, 3)
            3D velocities of each segment COM
            
        Returns
        -------
        momentum : array (3,)
            Total linear momentum vector
        """
        return np.sum(
            segment_velocities * np.array([seg.mass for seg in self.segments])[:, None],
            axis=0
        )
    
    def compute_oscillatory_energy(self, segment_angles: np.ndarray,
                                  segment_angular_velocities: np.ndarray) -> Dict:
        """
        Compute oscillatory energy at each segment.
        
        Energy = 0.5 * I * ω² + 0.5 * k * θ²
        
        Parameters
        ----------
        segment_angles : array (n_segments,)
            Angular positions
        segment_angular_velocities : array (n_segments,)
            Angular velocities
            
        Returns
        -------
        energy : dict
            Dictionary with kinetic, potential, and total energy per segment
        """
        kinetic = np.zeros(self.n_segments)
        potential = np.zeros(self.n_segments)
        
        for i, seg in enumerate(self.segments):
            I = seg.moment_of_inertia()
            omega = segment_angular_velocities[i]
            theta = segment_angles[i]
            
            # Kinetic energy
            kinetic[i] = 0.5 * I * omega**2
            
            # Potential energy (elastic, treating as torsional spring)
            # k = I * ω_n²
            k = I * (2 * np.pi * seg.natural_frequency)**2
            potential[i] = 0.5 * k * theta**2
        
        return {
            'kinetic': kinetic,
            'potential': potential,
            'total': kinetic + potential
        }
    
    def compute_coupling_torques(self, segment_angles: np.ndarray,
                                segment_angular_velocities: np.ndarray) -> np.ndarray:
        """
        Compute coupling torques between segments due to oscillatory coupling.
        
        τ_coupling_i = Σ_j C_ij * k_j * (θ_j - θ_i)
        
        Parameters
        ----------
        segment_angles : array (n_segments,)
            Current angular positions
        segment_angular_velocities : array (n_segments,)
            Current angular velocities
            
        Returns
        -------
        coupling_torques : array (n_segments,)
            Coupling torque at each segment
        """
        coupling_torques = np.zeros(self.n_segments)
        
        for i, seg_i in enumerate(self.segments):
            I_i = seg_i.moment_of_inertia()
            k_i = I_i * (2 * np.pi * seg_i.natural_frequency)**2
            
            for j, seg_j in enumerate(self.segments):
                if i != j:
                    I_j = seg_j.moment_of_inertia()
                    k_j = I_j * (2 * np.pi * seg_j.natural_frequency)**2
                    
                    # Coupling torque from segment j to segment i
                    coupling_torques[i] += (
                        self.segment_coupling[i, j] * k_j * 
                        (segment_angles[j] - segment_angles[i])
                    )
        
        return coupling_torques
    
    def equations_of_motion(self, t: float, state: np.ndarray,
                           external_torques: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute derivatives for coupled oscillator dynamics.
        
        Equations:
        θ̈_i = (1/I_i) * [τ_external_i + τ_coupling_i - c_i * θ̇_i]
        
        Parameters
        ----------
        t : float
            Current time
        state : array (2 * n_segments,)
            State vector [θ_1, ..., θ_n, ω_1, ..., ω_n]
        external_torques : array (n_segments,), optional
            External torques applied to each segment
            
        Returns
        -------
        dstate : array (2 * n_segments,)
            Time derivatives of state
        """
        # Unpack state
        theta = state[:self.n_segments]
        omega = state[self.n_segments:]
        
        # Compute coupling torques
        tau_coupling = self.compute_coupling_torques(theta, omega)
        
        # External torques
        if external_torques is None:
            tau_ext = np.zeros(self.n_segments)
        else:
            tau_ext = external_torques
        
        # Compute accelerations
        alpha = np.zeros(self.n_segments)
        for i, seg in enumerate(self.segments):
            I = seg.moment_of_inertia()
            
            # Damping torque
            omega_n = 2 * np.pi * seg.natural_frequency
            c = 2 * seg.damping_ratio * np.sqrt(I * I * omega_n**2)
            tau_damp = -c * omega[i]
            
            # Total torque
            tau_total = tau_ext[i] + tau_coupling[i] + tau_damp
            
            # Acceleration
            alpha[i] = tau_total / I
        
        # Pack derivatives
        dstate = np.concatenate([omega, alpha])
        
        return dstate
    
    def simulate_coupled_motion(self, initial_angles: np.ndarray,
                               initial_velocities: np.ndarray,
                               external_torque_func,
                               t_span: Tuple[float, float] = (0, 5.0),
                               dt: float = 0.01) -> Dict:
        """
        Simulate coupled oscillatory motion of kinematic chain.
        
        Parameters
        ----------
        initial_angles : array (n_segments,)
            Initial angular positions
        initial_velocities : array (n_segments,)
            Initial angular velocities
        external_torque_func : callable
            Function(t, theta, omega) -> torques
        t_span : tuple
            (start_time, end_time)
        dt : float
            Time step
            
        Returns
        -------
        results : dict
            Simulation results
        """
        from scipy.integrate import odeint
        
        # Initial state
        state0 = np.concatenate([initial_angles, initial_velocities])
        
        # Time array
        t_array = np.arange(t_span[0], t_span[1], dt)
        
        # Integrate
        def dynamics(state, t):
            theta = state[:self.n_segments]
            omega = state[self.n_segments:]
            tau_ext = external_torque_func(t, theta, omega)
            return self.equations_of_motion(t, state, tau_ext)
        
        states = odeint(dynamics, state0, t_array)
        
        # Extract results
        angles = states[:, :self.n_segments]
        velocities = states[:, self.n_segments:]
        
        # Compute energy over time
        energies = []
        for i in range(len(t_array)):
            energy = self.compute_oscillatory_energy(angles[i], velocities[i])
            energies.append(energy['total'])
        energies = np.array(energies)
        
        results = {
            'time': t_array,
            'angles': angles,
            'angular_velocities': velocities,
            'energies': energies,
            'coupling_matrix': self.segment_coupling
        }
        
        return results


class LowerLimbModel:
    """
    Complete lower limb model with oscillatory coupling.
    
    Combines body segments (thigh, shank, foot) with muscle models
    to simulate coordinated locomotion.
    """
    
    def __init__(self, body_mass: float, height: float):
        """
        Initialize lower limb model.
        
        Parameters
        ----------
        body_mass : float
            Body mass (kg)
        height : float
            Body height (m)
        """
        self.body_mass = body_mass
        self.height = height
        
        # Estimate segment lengths (anthropometric)
        self.thigh_length = 0.245 * height
        self.shank_length = 0.246 * height
        self.foot_length = 0.152 * height
        
        # Create segments
        self.thigh = BodySegmentParameters.create_segment(
            'thigh', body_mass, self.thigh_length
        )
        self.shank = BodySegmentParameters.create_segment(
            'shank', body_mass, self.shank_length
        )
        self.foot = BodySegmentParameters.create_segment(
            'foot', body_mass, self.foot_length
        )
        
        # Create kinematic chain
        self.chain = OscillatoryKinematicChain([self.thigh, self.shank, self.foot])
    
    def simulate_gait_cycle(self, stride_frequency: float = 1.5,
                           t_span: Tuple[float, float] = (0, 2.0),
                           dt: float = 0.001) -> Dict:
        """
        Simulate one gait cycle with oscillatory coupling.
        
        Parameters
        ----------
        stride_frequency : float
            Stride frequency (Hz, steps per second)
        t_span : tuple
            Time span
        dt : float
            Time step
            
        Returns
        -------
        results : dict
            Gait simulation results
        """
        # Initial conditions (mid-stance)
        initial_angles = np.array([30, -45, 15]) * np.pi / 180  # Hip, knee, ankle
        initial_velocities = np.zeros(3)
        
        # External torques (simplified muscle activation)
        def muscle_torques(t, theta, omega):
            # Simplified rhythmic muscle activation
            # Hip extensors
            tau_hip = 80 * np.sin(2 * np.pi * stride_frequency * t)
            
            # Knee extensors
            tau_knee = 50 * np.sin(2 * np.pi * stride_frequency * t + np.pi/4)
            
            # Ankle plantarflexors
            tau_ankle = 40 * np.sin(2 * np.pi * stride_frequency * t + np.pi/2)
            
            return np.array([tau_hip, tau_knee, tau_ankle])
        
        # Simulate
        results = self.chain.simulate_coupled_motion(
            initial_angles, initial_velocities,
            muscle_torques, t_span, dt
        )
        
        # Add joint names
        results['joint_names'] = ['Hip', 'Knee', 'Ankle']
        
        return results


def example_lower_limb_simulation():
    """Example: Simulate lower limb with oscillatory coupling during gait."""
    import matplotlib.pyplot as plt
    
    # Create model for 70 kg, 1.75 m person
    model = LowerLimbModel(body_mass=70, height=1.75)
    
    print("Simulating gait cycle with oscillatory coupling...")
    print(f"Thigh length: {model.thigh_length:.3f} m")
    print(f"Shank length: {model.shank_length:.3f} m")
    print(f"Foot length: {model.foot_length:.3f} m")
    
    # Simulate
    results = model.simulate_gait_cycle(stride_frequency=1.5, t_span=(0, 2.0))
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    t = results['time']
    angles = results['angles'] * 180 / np.pi  # Convert to degrees
    velocities = results['angular_velocities'] * 180 / np.pi
    
    # Joint angles
    for i, joint in enumerate(results['joint_names']):
        axes[0, 0].plot(t, angles[:, i], label=joint)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle (deg)')
    axes[0, 0].set_title('Joint Angles')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Joint angular velocities
    for i, joint in enumerate(results['joint_names']):
        axes[0, 1].plot(t, velocities[:, i], label=joint)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angular Velocity (deg/s)')
    axes[0, 1].set_title('Joint Angular Velocities')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Oscillatory energy
    energies = results['energies']
    for i, joint in enumerate(results['joint_names']):
        axes[1, 0].plot(t, energies[:, i], label=joint)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Energy (J)')
    axes[1, 0].set_title('Oscillatory Energy per Segment')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Coupling matrix
    coupling = results['coupling_matrix']
    im = axes[1, 1].imshow(coupling, cmap='hot', aspect='auto')
    axes[1, 1].set_title('Segment Coupling Matrix')
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_xticklabels(results['joint_names'])
    axes[1, 1].set_yticklabels(results['joint_names'])
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('oscillatory_gait_simulation.png', dpi=150)
    plt.show()
    
    print("\nPlot saved as 'oscillatory_gait_simulation.png'")
    
    # Print segment properties
    print("\n=== Segment Properties ===")
    for seg in model.chain.segments:
        print(f"\n{seg.name.upper()}:")
        print(f"  Mass: {seg.mass:.3f} kg")
        print(f"  Length: {seg.length:.3f} m")
        print(f"  Natural frequency: {seg.natural_frequency:.1f} Hz")
        print(f"  Moment of inertia: {seg.moment_of_inertia():.6f} kg⋅m²")


if __name__ == "__main__":
    example_lower_limb_simulation()

