"""
main_simulation.py

Complete integrated simulation of Maxwell's demon prisoner parable.
Combines all modules for comprehensive analysis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mechanics import PrisonerSystem
from thermodynamics import ThermodynamicsAnalyzer

def run_simulation(
    n_particles: int = 100,
    n_steps: int = 1000,
    demon_params: dict = None,
    verbose: bool = True
) -> tuple:
    """
    Run complete Maxwell demon simulation.

    Args:
        n_particles: Number of particles in system
        n_steps: Number of simulation steps
        demon_params: Parameters for Maxwell demon
        verbose: Print progress updates

    Returns:
        (system, analyzer) tuple
    """
    # Default demon parameters
    if demon_params is None:
        demon_params = {
            'information_capacity': 10.0,
            'selection_threshold': 1.0,
            'error_rate': 0.05,
            'memory_cost': 0.01
        }

    # Create system
    system = PrisonerSystem(
        n_particles=n_particles,
        initial_temperature=1.0,
        demon_params=demon_params
    )

    # Create analyzer
    analyzer = ThermodynamicsAnalyzer()

    if verbose:
        print("=" * 70)
        print("MAXWELL'S DEMON: THE PRISONER PARABLE SIMULATION")
        print("=" * 70)
        print()
        print(f"Configuration:")
        print(f"  Particles: {n_particles}")
        print(f"  Steps: {n_steps}")
        print(f"  Demon capacity: {demon_params['information_capacity']} bits")
        print(f"  Error rate: {demon_params['error_rate']:.1%}")
        print(f"  Memory cost: {demon_params['memory_cost']}")
        print()
        print(f"Initial state:")
        print(f"  Compartment A: {system.compartment_A.particle_count} particles, "
              f"T = {system.compartment_A.temperature:.3f}")
        print(f"  Compartment B: {system.compartment_B.particle_count} particles, "
              f"T = {system.compartment_B.temperature:.3f}")
        print()
        print("Running simulation...")

    # Run simulation
    for i in range(n_steps):
        system.step()
        analyzer.record_state(system)

        if verbose and (i + 1) % (n_steps // 10) == 0:
            progress = (i + 1) / n_steps * 100
            print(f"  {progress:5.1f}% | ΔT = {system.temperature_difference:6.3f} | "
                  f"S_total = {system.total_entropy:6.3f} | "
                  f"Accuracy = {system.demon.accuracy:5.1%}")

    if verbose:
        print()
        print("Simulation complete!")
        print()
        print(f"Final state:")
        print(f"  Compartment A: {system.compartment_A.particle_count} particles, "
              f"T = {system.compartment_A.temperature:.3f}")
        print(f"  Compartment B: {system.compartment_B.particle_count} particles, "
              f"T = {system.compartment_B.temperature:.3f}")
        print(f"  Temperature difference: {system.temperature_difference:.3f}")
        print(f"  Total entropy: {system.total_entropy:.3f}")
        print(f"  Demon processed: {system.demon.total_information_processed:.0f} bits")
        print(f"  Demon accuracy: {system.demon.accuracy:.2%}")
        print()
        print(analyzer.generate_report())

    return system, analyzer

def plot_results(system: PrisonerSystem, analyzer: ThermodynamicsAnalyzer):
    """Generate comprehensive plots of simulation results"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Maxwell's Demon: Prisoner Parable Results", fontsize=16, fontweight='bold')

    time = system.history['time']

    # 1. Temperature evolution
    ax = axes[0, 0]
    ax.plot(time, system.history['temp_A'], label='Compartment A', linewidth=2)
    ax.plot(time, system.history['temp_B'], label='Compartment B', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Entropy evolution
    ax = axes[0, 1]
    total_entropy = [a + b + c for a, b, c in zip(
        system.history['entropy_A'],
        system.history['entropy_B'],
        system.history['demon_entropy_cost']
    )]
    ax.plot(time, system.history['entropy_A'], label='Compartment A', alpha=0.7)
    ax.plot(time, system.history['entropy_B'], label='Compartment B', alpha=0.7)
    ax.plot(time, system.history['demon_entropy_cost'], label='Demon cost', alpha=0.7)
    ax.plot(time, total_entropy, label='Total', linewidth=2, color='black')
    ax.set_xlabel('Time')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Particle distribution
    ax = axes[1, 0]
    ax.plot(time, system.history['particles_A'], label='Compartment A', linewidth=2)
    ax.plot(time, system.history['particles_B'], label='Compartment B', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Particles')
    ax.set_title('Particle Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Demon information processing
    ax = axes[1, 1]
    ax.plot(time, system.history['demon_bits_processed'], linewidth=2, color='purple')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Bits Processed')
    ax.set_title('Demon Information Processing')
    ax.grid(True, alpha=0.3)

    # 5. Demon accuracy
    ax = axes[2, 0]
    ax.plot(time, system.history['demon_accuracy'], linewidth=2, color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Demon Performance')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)

    # 6. Temperature gradient vs entropy cost
    ax = axes[2, 1]
    temp_diff = [abs(a - b) for a, b in zip(
        system.history['temp_A'],
        system.history['temp_B']
    )]
    ax.plot(time, temp_diff, label='Temperature gradient', linewidth=2)
    ax.plot(time, system.history['demon_entropy_cost'],
            label='Demon entropy cost', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Gradient vs Information Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def main():
    """Run complete simulation with visualization"""
    # Run simulation
    system, analyzer = run_simulation(
        n_particles=200,
        n_steps=2000,
        demon_params={
            'information_capacity': 20.0,
            'selection_threshold': 1.0,
            'error_rate': 0.05,
            'memory_cost': 0.01
        },
        verbose=True
    )

    # Generate plots
    fig = plot_results(system, analyzer)
    plt.savefig('maxwell_demon_results.png', dpi=300, bbox_inches='tight')
    print()
    print("Results saved to 'maxwell_demon_results.png'")
    print()

    plt.show()

if __name__ == "__main__":
    main()
