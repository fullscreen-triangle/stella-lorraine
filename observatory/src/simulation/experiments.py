"""
experiments.py

Systematic parameter exploration for Maxwell demon simulation.
Tests different configurations to understand information-thermodynamics coupling.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict
from main_simulation import run_simulation
import pandas as pd

def experiment_error_rate_sweep(
    error_rates: List[float] = None,
    n_steps: int = 1000,
    n_particles: int = 100
) -> pd.DataFrame:
    """
    Sweep demon error rate and measure impact on performance.

    Tests: Does information quality affect thermodynamic outcomes?
    """
    if error_rates is None:
        error_rates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    print("=" * 70)
    print("EXPERIMENT 1: Error Rate Sweep")
    print("=" * 70)
    print()

    results = []

    for error_rate in error_rates:
        print(f"Testing error_rate = {error_rate:.2f}...")

        system, analyzer = run_simulation(
            n_particles=n_particles,
            n_steps=n_steps,
            demon_params={
                'information_capacity': 20.0,
                'selection_threshold': 1.0,
                'error_rate': error_rate,
                'memory_cost': 0.01
            },
            verbose=False
        )

        final_state = analyzer.states[-1]

        results.append({
            'error_rate': error_rate,
            'temperature_gradient': final_state.temperature_gradient,
            'total_entropy': final_state.total_entropy,
            'demon_accuracy': system.demon.accuracy,
            'bits_processed': system.demon.total_information_processed
        })

    df = pd.DataFrame(results)
    print()
    print(df.to_string(index=False))
    print()

    return df

def experiment_memory_cost_sweep(
    memory_costs: List[float] = None,
    n_steps: int = 1000,
    n_particles: int = 100
) -> pd.DataFrame:
    """
    Sweep demon memory cost and measure impact.

    Tests: Does Landauer's principle limit demon effectiveness?
    """
    if memory_costs is None:
        memory_costs = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]

    print("=" * 70)
    print("EXPERIMENT 2: Memory Cost Sweep")
    print("=" * 70)
    print()

    results = []

    for memory_cost in memory_costs:
        print(f"Testing memory_cost = {memory_cost:.3f}...")

        system, analyzer = run_simulation(
            n_particles=n_particles,
            n_steps=n_steps,
            demon_params={
                'information_capacity': 20.0,
                'selection_threshold': 1.0,
                'error_rate': 0.05,
                'memory_cost': memory_cost
            },
            verbose=False
        )

        final_state = analyzer.states[-1]

        results.append({
            'memory_cost': memory_cost,
            'temperature_gradient': final_state.temperature_gradient,
            'total_entropy': final_state.total_entropy,
            'demon_entropy_cost': final_state.demon_entropy_cost,
            'bits_processed': system.demon.total_information_processed
        })

    df = pd.DataFrame(results)
    print()
    print(df.to_string(index=False))
    print()

    return df

def experiment_capacity_sweep(
    capacities: List[float] = None,
    n_steps: int = 1000,
    n_particles: int = 100
) -> pd.DataFrame:
    """
    Sweep demon information capacity.

    Tests: Does processing power affect outcomes?
    """
    if capacities is None:
        capacities = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    print("=" * 70)
    print("EXPERIMENT 3: Information Capacity Sweep")
    print("=" * 70)
    print()

    results = []

    for capacity in capacities:
        print(f"Testing capacity = {capacity:.1f} bits...")

        system, analyzer = run_simulation(
            n_particles=n_particles,
            n_steps=n_steps,
            demon_params={
                'information_capacity': capacity,
                'selection_threshold': 1.0,
                'error_rate': 0.05,
                'memory_cost': 0.01
            },
            verbose=False
        )

        final_state = analyzer.states[-1]

        results.append({
            'capacity': capacity,
            'temperature_gradient': final_state.temperature_gradient,
            'total_entropy': final_state.total_entropy,
            'bits_processed': system.demon.total_information_processed
        })

    df = pd.DataFrame(results)
    print()
    print(df.to_string(index=False))
    print()

    return df

def plot_experiment_results(
    error_df: pd.DataFrame,
    memory_df: pd.DataFrame,
    capacity_df: pd.DataFrame
):
    """Plot all experiment results"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Maxwell Demon Parameter Sweep Results", fontsize=16, fontweight='bold')

    # Error rate experiments
    ax = axes[0, 0]
    ax.plot(error_df['error_rate'], error_df['temperature_gradient'], 'o-', linewidth=2)
    ax.set_xlabel('Error Rate')
    ax.set_ylabel('Temperature Gradient')
    ax.set_title('Error Rate vs Temperature Gradient')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(error_df['error_rate'], error_df['total_entropy'], 'o-', linewidth=2, color='red')
    ax.set_xlabel('Error Rate')
    ax.set_ylabel('Total Entropy')
    ax.set_title('Error Rate vs Total Entropy')
    ax.grid(True, alpha=0.3)

    # Memory cost experiments
    ax = axes[0, 1]
    ax.plot(memory_df['memory_cost'], memory_df['temperature_gradient'], 'o-', linewidth=2)
    ax.set_xlabel('Memory Cost')
    ax.set_ylabel('Temperature Gradient')
    ax.set_title('Memory Cost vs Temperature Gradient')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(memory_df['memory_cost'], memory_df['demon_entropy_cost'], 'o-', linewidth=2, color='purple')
    ax.set_xlabel('Memory Cost')
    ax.set_ylabel('Demon Entropy Cost')
    ax.set_title('Memory Cost vs Demon Entropy')
    ax.grid(True, alpha=0.3)

    # Capacity experiments
    ax = axes[0, 2]
    ax.plot(capacity_df['capacity'], capacity_df['temperature_gradient'], 'o-', linewidth=2)
    ax.set_xlabel('Information Capacity (bits)')
    ax.set_ylabel('Temperature Gradient')
    ax.set_title('Capacity vs Temperature Gradient')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(capacity_df['capacity'], capacity_df['bits_processed'], 'o-', linewidth=2, color='green')
    ax.set_xlabel('Information Capacity (bits)')
    ax.set_ylabel('Total Bits Processed')
    ax.set_title('Capacity vs Information Processed')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def main():
    """Run all experiments"""
    print()
    print("=" * 70)
    print("MAXWELL DEMON PARAMETER EXPLORATION")
    print("=" * 70)
    print()

    # Run experiments
    error_df = experiment_error_rate_sweep(n_steps=1000)
    memory_df = experiment_memory_cost_sweep(n_steps=1000)
    capacity_df = experiment_capacity_sweep(n_steps=1000)

    # Plot results
    fig = plot_experiment_results(error_df, memory_df, capacity_df)
    plt.savefig('parameter_sweep_results.png', dpi=300, bbox_inches='tight')
    print()
    print("Results saved to 'parameter_sweep_results.png'")
    print()

    plt.show()

if __name__ == "__main__":
    main()
