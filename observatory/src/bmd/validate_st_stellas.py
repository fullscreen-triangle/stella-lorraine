"""
validate_st_stellas.py

Complete validation of St-Stellas categorical dynamics framework
using Maxwell's demon prisoner parable.

Validates Theorem 3.12: BMD ≡ S-Navigation ≡ Categorical Completion
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime
import sys
sys.path.append('.')

from mechanics import PrisonerSystem
from thermodynamics import ThermodynamicsAnalyzer
from categorical_tracker import CategoricalTracker

def run_categorical_validation(
    n_particles: int = 200,
    n_steps: int = 2000,
    demon_params: dict = None,
    verbose: bool = True
):
    """
    Run Maxwell demon simulation with categorical tracking.

    This validates the St-Stellas framework by showing that:
    1. BMD operations = categorical completions
    2. S-distance minimization = optimal demon behavior
    3. Equivalence classes have high degeneracy (~10^6)
    """
    if demon_params is None:
        demon_params = {
            'information_capacity': 20.0,
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

    # Create analyzers
    thermo_analyzer = ThermodynamicsAnalyzer()
    categorical_tracker = CategoricalTracker(observable_precision=0.05)

    if verbose:
        print("=" * 70)
        print("ST-STELLAS CATEGORICAL DYNAMICS VALIDATION")
        print("Prisoner Parable with Categorical State Tracking")
        print("=" * 70)
        print()
        print(f"Configuration:")
        print(f"  Particles: {n_particles}")
        print(f"  Steps: {n_steps}")
        print(f"  Demon capacity: {demon_params['information_capacity']} bits")
        print(f"  Error rate: {demon_params['error_rate']:.1%}")
        print()
        print("Running simulation with categorical tracking...")
        print()

    # Run simulation with categorical tracking
    for i in range(n_steps):
        # Record pre-state
        pre_state = categorical_tracker.record_state(system, system.time)

        # System step (includes demon decisions)
        system.step()

        # Record post-state
        post_state = categorical_tracker.record_state(system, system.time)

        # Track BMD operations (approximate from demon statistics)
        if system.demon.decisions_made > 0:
            categorical_tracker.record_bmd_operation(
                particle_id=-1,  # Unknown in current implementation
                observed=True,
                classified_state='fast/slow',
                decision='allowed' if system.demon.accuracy > 0.5 else 'rejected',
                pre_state_idx=pre_state.index,
                post_state_idx=post_state.index
            )

        # Thermodynamic tracking
        thermo_analyzer.record_state(system)

        # Progress
        if verbose and (i + 1) % (n_steps // 10) == 0:
            progress = (i + 1) / n_steps * 100
            s_coords = categorical_tracker.s_trajectory[-1] if categorical_tracker.s_trajectory else (0,0,0)
            print(f"  {progress:5.1f}% | C_index = {categorical_tracker.current_index:5d} | "
                  f"S = ({s_coords[0]:6.2f}, {s_coords[1]:6.2f}, {s_coords[2]:6.2f}) | "
                  f"Classes = {len(categorical_tracker.equivalence_classes):3d}")

    if verbose:
        print()
        print("Simulation complete!")
        print()
        print(categorical_tracker.generate_report())
        print()
        print(thermo_analyzer.generate_report())

    return system, thermo_analyzer, categorical_tracker


def plot_categorical_results(
    system: PrisonerSystem,
    categorical_tracker: CategoricalTracker
):
    """Generate comprehensive plots showing categorical dynamics"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    fig.suptitle("St-Stellas Categorical Dynamics Validation\nMaxwell's Demon Prisoner Parable",
                 fontsize=14, fontweight='bold')

    # Extract data
    time = system.history['time']
    states = categorical_tracker.states
    state_times = [s.timestamp for s in states]

    # 1. Categorical completion trajectory
    ax1 = fig.add_subplot(gs[0, :])
    categorical_indices = [s.index for s in states]
    ax1.plot(state_times, categorical_indices, linewidth=2, color='blue')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Categorical Index $C_i$')
    ax1.set_title('Categorical Completion Sequence (Axiom 1: Irreversibility)')
    ax1.grid(True, alpha=0.3)

    # 2. Equivalence class degeneracy histogram
    ax2 = fig.add_subplot(gs[1, 0])
    degeneracies = [ec.degeneracy for ec in categorical_tracker.equivalence_classes.values()]
    ax2.hist(degeneracies, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('Equivalence Class Degeneracy $|[C]_\\sim|$')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Equivalence Class Distribution\nAvg: {np.mean(degeneracies):.1f}')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Information content per class
    ax3 = fig.add_subplot(gs[1, 1])
    info_contents = [ec.information_content for ec in categorical_tracker.equivalence_classes.values()]
    ax3.hist(info_contents, bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax3.set_xlabel('Information Content (bits)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Information per Equivalence Class\nAvg: {np.mean(info_contents):.2f} bits')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. BMD probability enhancement
    ax4 = fig.add_subplot(gs[1, 2])
    enhancement = categorical_tracker.compute_bmd_probability_enhancement()
    ax4.bar(['$p_{BMD}/p_0$'], [enhancement], color='red', alpha=0.7, width=0.5)
    ax4.axhline(y=1e6, color='green', linestyle='--', label='Mizraji min ($10^6$)')
    ax4.axhline(y=1e11, color='blue', linestyle='--', label='Mizraji max ($10^{11}$)')
    ax4.set_ylabel('Probability Enhancement')
    ax4.set_yscale('log')
    ax4.set_title('BMD Probability Enhancement\n(Mizraji 2021)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, which='both')

    # 5. S-space trajectory (3D projection)
    ax5 = fig.add_subplot(gs[2, :], projection='3d')
    s_traj = np.array(categorical_tracker.s_trajectory)
    if len(s_traj) > 0:
        ax5.plot(s_traj[:, 0], s_traj[:, 1], s_traj[:, 2],
                linewidth=2, color='orange', alpha=0.7)
        ax5.scatter(s_traj[0, 0], s_traj[0, 1], s_traj[0, 2],
                   color='green', s=100, label='Start', marker='o')
        ax5.scatter(s_traj[-1, 0], s_traj[-1, 1], s_traj[-1, 2],
                   color='red', s=100, label='End', marker='s')
        ax5.set_xlabel('$S_k$ (Knowledge)')
        ax5.set_ylabel('$S_t$ (Time)')
        ax5.set_zlabel('$S_e$ (Entropy)')
        ax5.set_title('S-Space Navigation Trajectory\n(Theorem 3.12: S-Navigation ≡ BMD)')
        ax5.legend()

    # 6. S-coordinate evolution
    ax6 = fig.add_subplot(gs[3, 0])
    if len(s_traj) > 0:
        ax6.plot(state_times, s_traj[:, 0], label='$S_k$ (Knowledge)', linewidth=2)
        ax6.plot(state_times, s_traj[:, 1], label='$S_t$ (Time)', linewidth=2)
        ax6.plot(state_times, s_traj[:, 2], label='$S_e$ (Entropy)', linewidth=2)
        ax6.set_xlabel('Time')
        ax6.set_ylabel('S-Coordinate Value')
        ax6.set_title('S-Coordinate Evolution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    # 7. Categorical completion rate
    ax7 = fig.add_subplot(gs[3, 1])
    # Compute local dC/dt
    window = 50
    completion_rates = []
    rate_times = []
    for i in range(window, len(states)):
        dt = states[i].timestamp - states[i-window].timestamp
        dC = window
        rate = dC / dt if dt > 0 else 0
        completion_rates.append(rate)
        rate_times.append(states[i].timestamp)

    if completion_rates:
        ax7.plot(rate_times, completion_rates, linewidth=2, color='brown')
        ax7.set_xlabel('Time')
        ax7.set_ylabel('$dC/dt$ (states/time)')
        ax7.set_title('Categorical Completion Rate\n(Fundamental Clock)')
        ax7.grid(True, alpha=0.3)

    # 8. Temperature vs Categorical States
    ax8 = fig.add_subplot(gs[3, 2])
    temp_A = [s.observables['temp_A'] for s in states]
    temp_B = [s.observables['temp_B'] for s in states]
    ax8.scatter(temp_A, temp_B, c=state_times, cmap='viridis',
               alpha=0.6, s=20)
    ax8.set_xlabel('Temperature A')
    ax8.set_ylabel('Temperature B')
    ax8.set_title('Temperature State Space\n(Equivalence Class Observable)')
    cbar = plt.colorbar(ax8.collections[0], ax=ax8)
    cbar.set_label('Time')
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Run complete categorical validation"""
    print()

    # Run simulation with categorical tracking
    system, thermo_analyzer, categorical_tracker = run_categorical_validation(
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
    print("Generating categorical dynamics visualizations...")
    fig = plot_categorical_results(system, categorical_tracker)
    plt.savefig('st_stellas_validation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Visualization saved to 'st_stellas_validation.png'")
    print()

    # Verification summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    verification = categorical_tracker.verify_st_stellas_equivalence()

    all_passed = all(verification.values())

    for test, passed in verification.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test}")

    print()
    if all_passed:
        print("✓✓✓ ST-STELLAS FRAMEWORK VALIDATED ✓✓✓")
        print("BMD operation ≡ S-Navigation ≡ Categorical Completion")
    else:
        print("⚠ Some tests failed - review parameters")
    print("=" * 70)
    print()

    # Helper to convert numpy types to native Python types
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_native(obj.tolist())
        else:
            return obj

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'validation_summary': {
            'all_passed': bool(all_passed),
            'tests': {k: bool(v) for k, v in verification.items()}
        },
        'categorical_metrics': {
            'total_states': len(categorical_tracker.states),
            'equivalence_classes': len(categorical_tracker.equivalence_classes),
            'categorical_completion_rate': float(categorical_tracker.compute_categorical_completion_rate()),
            'total_information_processed': float(categorical_tracker.compute_total_information_processed()),
            'bmd_probability_enhancement': float(categorical_tracker.compute_bmd_probability_enhancement())
        },
        'demon_performance': {
            'accuracy': float(system.demon.accuracy),
            'bits_processed': float(system.demon.total_information_processed),
            'entropy_cost': float(system.demon.entropy_cost)
        },
        'thermodynamics': {
            'final_temp_A': float(system.compartment_A.temperature),
            'final_temp_B': float(system.compartment_B.temperature),
            'temp_difference': float(system.temperature_difference),
            'total_entropy': float(system.total_entropy)
        }
    }

    results_file = f'st_stellas_validation_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to '{results_file}'")
    print()

if __name__ == "__main__":
    main()
