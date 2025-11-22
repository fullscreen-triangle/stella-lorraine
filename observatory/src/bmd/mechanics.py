"""
prisoner_core.py

Core implementation of Maxwell's demon (prisoner) with information processing.
Based on Mizraji (2021) biological Maxwell demons framework.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class ParticleState(Enum):
    """Particle energy states"""
    SLOW = 0
    FAST = 1

@dataclass
class Particle:
    """Individual particle in the system"""
    velocity: float
    position: np.ndarray
    state: ParticleState
    id: int

    @property
    def kinetic_energy(self) -> float:
        """Calculate kinetic energy"""
        return 0.5 * self.velocity**2

    @property
    def temperature_contribution(self) -> float:
        """Temperature is proportional to mean kinetic energy"""
        return self.kinetic_energy

@dataclass
class Compartment:
    """One side of the divided system"""
    particles: List[Particle]
    volume: float
    name: str

    @property
    def temperature(self) -> float:
        """Mean temperature of compartment"""
        if not self.particles:
            return 0.0
        return np.mean([p.temperature_contribution for p in self.particles])

    @property
    def entropy(self) -> float:
        """Statistical entropy (simplified)"""
        if not self.particles:
            return 0.0
        # S = k * ln(W) where W is number of microstates
        # Approximated by velocity distribution width
        velocities = [p.velocity for p in self.particles]
        if len(velocities) < 2:
            return 0.0
        return np.std(velocities)  # Simplified entropy measure

    @property
    def particle_count(self) -> int:
        return len(self.particles)

class MaxwellDemon:
    """
    The Prisoner: Information processor that selects inputs and directs outputs.

    Implements Mizraji's biological Maxwell demon:
    - Selects which particles to allow through
    - Directs particles to specific compartments
    - Acts as information catalyst
    """

    def __init__(
        self,
        information_capacity: float = 1.0,
        selection_threshold: float = 0.5,
        error_rate: float = 0.0,
        memory_cost: float = 0.0
    ):
        """
        Args:
            information_capacity: Bits of information demon can process
            selection_threshold: Velocity threshold for fast/slow classification
            error_rate: Probability of misclassification (0-1)
            memory_cost: Thermodynamic cost per bit of information stored
        """
        self.information_capacity = information_capacity
        self.selection_threshold = selection_threshold
        self.error_rate = error_rate
        self.memory_cost = memory_cost

        # State tracking
        self.bits_processed = 0.0
        self.decisions_made = 0
        self.correct_decisions = 0
        self.entropy_cost = 0.0

    def observe_particle(self, particle: Particle) -> bool:
        """
        Observe particle and determine if it's fast or slow.

        This is the INFORMATION ACQUISITION step.
        Returns True if observation successful, False if capacity exceeded.
        """
        # Check if demon has capacity to process
        if self.bits_processed >= self.information_capacity:
            return False

        # Information cost: 1 bit per observation
        self.bits_processed += 1.0

        # Entropy cost of measurement (Landauer's principle: kT ln(2) per bit)
        self.entropy_cost += self.memory_cost

        return True

    def classify_particle(self, particle: Particle) -> ParticleState:
        """
        Classify particle as FAST or SLOW.

        This is the INFORMATION PROCESSING step.
        Subject to error_rate for realistic implementation.
        """
        # True classification
        true_state = (ParticleState.FAST if particle.velocity > self.selection_threshold
                     else ParticleState.SLOW)

        # Apply error rate
        if np.random.random() < self.error_rate:
            # Misclassification
            classified_state = (ParticleState.SLOW if true_state == ParticleState.FAST
                              else ParticleState.FAST)
        else:
            # Correct classification
            classified_state = true_state
            self.correct_decisions += 1

        self.decisions_made += 1
        return classified_state

    def decide_action(
        self,
        particle: Particle,
        source_compartment: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide whether to allow particle through gate and where to direct it.

        This is the INFORMATION CATALYSIS step:
        - Selection: Choose which particles to process
        - Direction: Target specific compartments

        Returns:
            (allow_through, target_compartment)
        """
        # Observe particle
        if not self.observe_particle(particle):
            return False, None

        # Classify particle
        state = self.classify_particle(particle)

        # MAXWELL'S DEMON STRATEGY:
        # Allow fast particles from A to B
        # Allow slow particles from B to A
        # This creates temperature gradient without doing work!

        if source_compartment == "A":
            if state == ParticleState.FAST:
                return True, "B"
            else:
                return False, None
        else:  # source_compartment == "B"
            if state == ParticleState.SLOW:
                return True, "A"
            else:
                return False, None

    def reset_memory(self):
        """
        Erase memory to make room for new information.

        This is where LANDAUER'S PRINCIPLE applies:
        Erasing information costs entropy.
        """
        # Entropy cost of erasure
        erasure_cost = self.bits_processed * self.memory_cost
        self.entropy_cost += erasure_cost

        # Reset counters
        self.bits_processed = 0.0

    @property
    def accuracy(self) -> float:
        """Classification accuracy"""
        if self.decisions_made == 0:
            return 0.0
        return self.correct_decisions / self.decisions_made

    @property
    def total_information_processed(self) -> float:
        """Total bits processed (lifetime)"""
        return float(self.decisions_made)

class PrisonerSystem:
    """
    Complete prisoner parable system.

    Two compartments (A and B) separated by gate.
    Maxwell demon controls gate based on particle velocities.
    """

    def __init__(
        self,
        n_particles: int = 100,
        initial_temperature: float = 1.0,
        demon_params: Optional[dict] = None
    ):
        """
        Args:
            n_particles: Total number of particles
            initial_temperature: Initial system temperature
            demon_params: Parameters for Maxwell demon
        """
        self.n_particles = n_particles
        self.initial_temperature = initial_temperature

        # Create demon
        demon_params = demon_params or {}
        self.demon = MaxwellDemon(**demon_params)

        # Initialize compartments
        self.compartment_A = Compartment(particles=[], volume=1.0, name="A")
        self.compartment_B = Compartment(particles=[], volume=1.0, name="B")

        # Initialize particles
        self._initialize_particles()

        # Time tracking
        self.time = 0.0
        self.dt = 0.01

        # History for analysis
        self.history = {
            'time': [],
            'temp_A': [],
            'temp_B': [],
            'entropy_A': [],
            'entropy_B': [],
            'particles_A': [],
            'particles_B': [],
            'demon_entropy_cost': [],
            'demon_bits_processed': [],
            'demon_accuracy': []
        }

    def _initialize_particles(self):
        """Initialize particles with Maxwell-Boltzmann distribution"""
        # Generate velocities from Maxwell-Boltzmann distribution
        velocities = np.random.normal(
            loc=self.initial_temperature,
            scale=np.sqrt(self.initial_temperature),
            size=self.n_particles
        )
        velocities = np.abs(velocities)  # Only positive velocities

        # Create particles and distribute randomly
        for i, v in enumerate(velocities):
            particle = Particle(
                velocity=v,
                position=np.random.rand(3),
                state=ParticleState.SLOW,
                id=i
            )

            # Randomly assign to compartment
            if np.random.random() < 0.5:
                self.compartment_A.particles.append(particle)
            else:
                self.compartment_B.particles.append(particle)

    def step(self):
        """Single time step of simulation"""
        # Particles attempt to cross gate
        self._attempt_crossings()

        # Update time
        self.time += self.dt

        # Record history
        self._record_state()

    def _attempt_crossings(self):
        """Particles attempt to cross between compartments"""
        # Probability of attempting to cross (proportional to velocity)

        # A -> B attempts
        for particle in list(self.compartment_A.particles):
            attempt_prob = particle.velocity * self.dt
            if np.random.random() < attempt_prob:
                allow, target = self.demon.decide_action(particle, "A")
                if allow and target == "B":
                    self.compartment_A.particles.remove(particle)
                    self.compartment_B.particles.append(particle)

        # B -> A attempts
        for particle in list(self.compartment_B.particles):
            attempt_prob = particle.velocity * self.dt
            if np.random.random() < attempt_prob:
                allow, target = self.demon.decide_action(particle, "B")
                if allow and target == "A":
                    self.compartment_B.particles.remove(particle)
                    self.compartment_A.particles.append(particle)

        # Demon may need to reset memory periodically
        if self.demon.bits_processed >= self.demon.information_capacity:
            self.demon.reset_memory()

    def _record_state(self):
        """Record current state for analysis"""
        self.history['time'].append(self.time)
        self.history['temp_A'].append(self.compartment_A.temperature)
        self.history['temp_B'].append(self.compartment_B.temperature)
        self.history['entropy_A'].append(self.compartment_A.entropy)
        self.history['entropy_B'].append(self.compartment_B.entropy)
        self.history['particles_A'].append(self.compartment_A.particle_count)
        self.history['particles_B'].append(self.compartment_B.particle_count)
        self.history['demon_entropy_cost'].append(self.demon.entropy_cost)
        self.history['demon_bits_processed'].append(self.demon.total_information_processed)
        self.history['demon_accuracy'].append(self.demon.accuracy)

    @property
    def total_entropy(self) -> float:
        """Total system entropy including demon's cost"""
        return (self.compartment_A.entropy +
                self.compartment_B.entropy +
                self.demon.entropy_cost)

    @property
    def temperature_difference(self) -> float:
        """Temperature difference between compartments"""
        return abs(self.compartment_A.temperature - self.compartment_B.temperature)

def main():
    """Test the prisoner system"""
    print("=" * 60)
    print("MAXWELL'S DEMON: THE PRISONER PARABLE")
    print("=" * 60)
    print()

    # Create system
    system = PrisonerSystem(
        n_particles=100,
        initial_temperature=1.0,
        demon_params={
            'information_capacity': 10.0,
            'selection_threshold': 1.0,
            'error_rate': 0.1,
            'memory_cost': 0.01
        }
    )

    print(f"Initial state:")
    print(f"  Compartment A: {system.compartment_A.particle_count} particles, T = {system.compartment_A.temperature:.3f}")
    print(f"  Compartment B: {system.compartment_B.particle_count} particles, T = {system.compartment_B.temperature:.3f}")
    print(f"  Total entropy: {system.total_entropy:.3f}")
    print()

    # Run simulation
    n_steps = 1000
    print(f"Running {n_steps} steps...")

    for i in range(n_steps):
        system.step()

        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}: ΔT = {system.temperature_difference:.3f}, "
                  f"S_total = {system.total_entropy:.3f}, "
                  f"Demon accuracy = {system.demon.accuracy:.2%}")

    print()
    print("Final state:")
    print(f"  Compartment A: {system.compartment_A.particle_count} particles, T = {system.compartment_A.temperature:.3f}")
    print(f"  Compartment B: {system.compartment_B.particle_count} particles, T = {system.compartment_B.temperature:.3f}")
    print(f"  Temperature difference: {system.temperature_difference:.3f}")
    print(f"  Total entropy: {system.total_entropy:.3f}")
    print(f"  Demon processed: {system.demon.total_information_processed:.0f} bits")
    print(f"  Demon accuracy: {system.demon.accuracy:.2%}")
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
