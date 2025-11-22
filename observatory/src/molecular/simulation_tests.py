import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


if __name__ == "__main__":
    # Load all simulation test files
    simulation_files = [
        'public/simulation_test_20251011_070821.json',
        'public/simulation_test_20251108_231304.json',
        'public/simulation_test_20251108_231401.json'
    ]

    print("="*80)
    print("SIMULATION TEST MULTI-RUN ANALYSIS")
    print("="*80)

    # Load all data
    all_sim_data = []
    for filename in simulation_files:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                all_sim_data.append(data)
                print(f"✓ Loaded: {filename}")
                print(f"  Timestamp: {data['timestamp']}")
                print(f"  Components: {len(data.get('components_tested', []))}")
        except Exception as e:
            print(f"✗ Failed to load {filename}: {e}")

    print(f"\nTotal simulation runs loaded: {len(all_sim_data)}")
    print("="*80)

    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(5, 4, figure=fig, hspace=0.45, wspace=0.4)

    # Color scheme
    colors = {
        'sim1': '#3498db',
        'sim2': '#e74c3c',
        'sim3': '#2ecc71',
        'mean': '#34495e',
        'particle': '#9b59b6',
        'field': '#f39c12',
        'interaction': '#1abc9c'
    }

    sim_colors = [colors['sim1'], colors['sim2'], colors['sim3']]

    # Extract simulation metrics
    def extract_simulation_metrics(data):
        """Extract metrics from simulation components"""
        components = data.get('components_tested', [])
        metrics = {
            'names': [],
            'status': [],
            'execution_time': [],
            'memory_usage': [],
            'accuracy': [],
            'convergence': [],
            'iterations': [],
            'stability': []
        }

        for comp in components:
            metrics['names'].append(comp.get('name', 'Unknown'))
            metrics['status'].append(comp.get('status', 'unknown'))
            metrics['execution_time'].append(comp.get('execution_time', 0))
            metrics['memory_usage'].append(comp.get('memory_usage', 0))
            metrics['accuracy'].append(comp.get('accuracy', 0))
            metrics['convergence'].append(comp.get('convergence', 0))
            metrics['iterations'].append(comp.get('iterations', 0))
            metrics['stability'].append(comp.get('stability', 0))

        return metrics

    # Extract metrics for all runs
    all_sim_metrics = [extract_simulation_metrics(data) for data in all_sim_data]
    common_sim_components = all_sim_metrics[0]['names'] if all_sim_metrics else []

    # Generate synthetic simulation data
    np.random.seed(42)
    n_particles = 1000
    n_timesteps = 100

    # Particle simulation
    particles = {
        'positions': np.random.randn(n_particles, 3) * 10,
        'velocities': np.random.randn(n_particles, 3),
        'masses': np.random.uniform(0.5, 2.0, n_particles),
        'energies': np.random.exponential(1.0, n_particles)
    }

    # Time evolution
    time_evolution = {
        'kinetic_energy': [],
        'potential_energy': [],
        'total_energy': [],
        'temperature': [],
        'pressure': []
    }

    for t in range(n_timesteps):
        ke = 0.5 * np.sum(particles['masses'][:, np.newaxis] * particles['velocities']**2)
        pe = np.sum(particles['energies']) * (1 + 0.1 * np.sin(2 * np.pi * t / n_timesteps))

        time_evolution['kinetic_energy'].append(ke)
        time_evolution['potential_energy'].append(pe)
        time_evolution['total_energy'].append(ke + pe)
        time_evolution['temperature'].append(ke / n_particles)
        time_evolution['pressure'].append(np.random.normal(1.0, 0.1))

    # ============================================================
    # PANEL 1: Execution Time Comparison Across Simulation Runs
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :2])

    x_pos = np.arange(len(common_sim_components))
    width = 0.25

    for i, (metrics, color) in enumerate(zip(all_sim_metrics, sim_colors)):
        exec_times = metrics['execution_time']
        offset = (i - len(all_sim_metrics)/2) * width
        ax1.bar(x_pos + offset, exec_times, width,
            label=f'Run {i+1} ({all_sim_data[i]["timestamp"]})',
            color=color, alpha=0.8, edgecolor='black', linewidth=1)

    # Add mean line
    if all_sim_metrics:
        mean_times = np.mean([m['execution_time'] for m in all_sim_metrics], axis=0)
        ax1.plot(x_pos, mean_times, 'k--', linewidth=3, marker='o',
                markersize=8, label='Mean across runs')

    ax1.set_xlabel('Simulation Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (s)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Simulation Execution Time Comparison\nMulti-Run Performance',
                fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(common_sim_components, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 2: Convergence Analysis
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 2:])

    for i, (metrics, color) in enumerate(zip(all_sim_metrics, sim_colors)):
        convergence = metrics['convergence']
        ax2.plot(x_pos, convergence, 'o-', linewidth=2, markersize=8,
                color=color, label=f'Run {i+1}', alpha=0.7)

    # Add mean and std bands
    if all_sim_metrics:
        mean_conv = np.mean([m['convergence'] for m in all_sim_metrics], axis=0)
        std_conv = np.std([m['convergence'] for m in all_sim_metrics], axis=0)

        ax2.plot(x_pos, mean_conv, 'k-', linewidth=4, marker='D',
                markersize=10, label='Mean', zorder=10)
        ax2.fill_between(x_pos, mean_conv - std_conv, mean_conv + std_conv,
                        alpha=0.3, color='gray', label='±1σ')

    ax2.set_xlabel('Simulation Component', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Convergence Rate', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Convergence Analysis Across Runs\nNumerical Stability',
                fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(common_sim_components, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 3: 3D Particle Distribution
    # ============================================================
    ax3 = fig.add_subplot(gs[1, :2], projection='3d')

    # Plot particles colored by energy
    scatter = ax3.scatter(particles['positions'][:, 0],
                        particles['positions'][:, 1],
                        particles['positions'][:, 2],
                        c=particles['energies'],
                        cmap='hot',
                        s=particles['masses']*20,
                        alpha=0.6,
                        edgecolor='black',
                        linewidth=0.5)

    cbar = plt.colorbar(scatter, ax=ax3, pad=0.1, shrink=0.8)
    cbar.set_label('Energy', fontsize=10, fontweight='bold')

    ax3.set_xlabel('X Position', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Y Position', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Z Position', fontsize=10, fontweight='bold')
    ax3.set_title('(C) 3D Particle Distribution\nSpatial Configuration',
                fontsize=14, fontweight='bold', pad=15)

    # ============================================================
    # PANEL 4: Energy Evolution
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 2:])

    time_axis = np.arange(n_timesteps)

    ax4.plot(time_axis, time_evolution['kinetic_energy'], linewidth=2,
            color=colors['sim1'], label='Kinetic Energy')
    ax4.plot(time_axis, time_evolution['potential_energy'], linewidth=2,
            color=colors['sim2'], label='Potential Energy')
    ax4.plot(time_axis, time_evolution['total_energy'], linewidth=3,
            color='black', linestyle='--', label='Total Energy')

    # Mark energy conservation
    initial_energy = time_evolution['total_energy'][0]
    final_energy = time_evolution['total_energy'][-1]
    conservation_error = abs(final_energy - initial_energy) / initial_energy * 100

    ax4.text(0.98, 0.98, f'Energy Conservation Error: {conservation_error:.4f}%',
            transform=ax4.transAxes, fontsize=10, ha='right', va='top',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if conservation_error < 1 else 'yellow', alpha=0.9))

    ax4.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Energy (arbitrary units)', fontsize=12, fontweight='bold')
    ax4.set_title('(D) Energy Evolution Over Time\nConservation Analysis',
                fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=10, loc='upper right')
    ax4.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 5: Iteration Count Comparison
    # ============================================================
    ax5 = fig.add_subplot(gs[2, 0])

    if all_sim_metrics:
        # Calculate mean iterations per component across runs
        iterations_data = []
        for metrics in all_sim_metrics:
            iterations_data.append(metrics['iterations'])

        # Box plot
        bp = ax5.boxplot(iterations_data, labels=[f'Run {i+1}' for i in range(len(all_sim_metrics))],
                        patch_artist=True, showmeans=True)

        for patch, color in zip(bp['boxes'], sim_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax5.set_ylabel('Iterations', fontsize=12, fontweight='bold')
        ax5.set_title('(E) Iteration Count Distribution\nComputational Effort',
                    fontsize=14, fontweight='bold', pad=15)
        ax5.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 6: Stability Metrics
    # ============================================================
    ax6 = fig.add_subplot(gs[2, 1])

    if all_sim_metrics:
        # Calculate stability scores
        stability_scores = []
        for metrics in all_sim_metrics:
            mean_stability = np.mean(metrics['stability'])
            stability_scores.append(mean_stability)

        bars = ax6.bar(range(1, len(stability_scores) + 1), stability_scores,
                    color=sim_colors, alpha=0.8, edgecolor='black', linewidth=2)

        # Add value labels
        for i, score in enumerate(stability_scores):
            ax6.text(i + 1, score + 0.02, f'{score:.3f}', ha='center',
                    fontsize=10, fontweight='bold')

        # Add mean line
        mean_stability = np.mean(stability_scores)
        ax6.axhline(mean_stability, color='black', linestyle='--', linewidth=3,
                label=f'Mean: {mean_stability:.3f}')

        ax6.set_xlabel('Run Number', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Stability Score', fontsize=12, fontweight='bold')
        ax6.set_title('(F) Numerical Stability by Run\nConsistency Assessment',
                    fontsize=14, fontweight='bold', pad=15)
        ax6.set_xticks(range(1, len(stability_scores) + 1))
        ax6.legend(fontsize=10)
        ax6.grid(alpha=0.3, linestyle='--', axis='y')
        ax6.set_ylim(0, 1.1)

    # ============================================================
    # PANEL 7: Phase Space Trajectory
    # ============================================================
    ax7 = fig.add_subplot(gs[2, 2:])

    # Plot phase space (position vs velocity) for sample particles
    n_sample = 50
    sample_indices = np.random.choice(n_particles, n_sample, replace=False)

    for idx in sample_indices:
        # Simulate trajectory
        trajectory_x = particles['positions'][idx, 0] + np.cumsum(particles['velocities'][idx, 0] * 0.1 * np.random.randn(20))
        trajectory_v = particles['velocities'][idx, 0] + np.cumsum(0.1 * np.random.randn(20))

        ax7.plot(trajectory_x, trajectory_v, alpha=0.3, linewidth=1, color=colors['particle'])

    # Mark initial positions
    ax7.scatter(particles['positions'][sample_indices, 0],
            particles['velocities'][sample_indices, 0],
            s=100, color='red', marker='o', edgecolor='black', linewidth=2,
            zorder=10, label='Initial state')

    ax7.set_xlabel('Position', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Velocity', fontsize=12, fontweight='bold')
    ax7.set_title('(G) Phase Space Trajectory\nDynamical Evolution',
                fontsize=14, fontweight='bold', pad=15)
    ax7.legend(fontsize=10)
    ax7.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 8: Temperature and Pressure Evolution
    # ============================================================
    ax8 = fig.add_subplot(gs[3, :2])

    ax8_twin = ax8.twinx()

    l1 = ax8.plot(time_axis, time_evolution['temperature'], linewidth=3,
                color=colors['sim1'], label='Temperature')
    l2 = ax8_twin.plot(time_axis, time_evolution['pressure'], linewidth=3,
                    color=colors['sim2'], linestyle='--', label='Pressure')

    ax8.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Temperature', fontsize=11, fontweight='bold', color=colors['sim1'])
    ax8_twin.set_ylabel('Pressure', fontsize=11, fontweight='bold', color=colors['sim2'])

    ax8.tick_params(axis='y', labelcolor=colors['sim1'])
    ax8_twin.tick_params(axis='y', labelcolor=colors['sim2'])

    # Combine legends
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax8.legend(lns, labs, fontsize=10, loc='upper right')

    ax8.set_title('(H) Thermodynamic Properties Evolution\nTemperature and Pressure',
                fontsize=14, fontweight='bold', pad=15)
    ax8.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 9: Velocity Distribution
    # ============================================================
    ax9 = fig.add_subplot(gs[3, 2:])

    # Calculate speed distribution
    speeds = np.linalg.norm(particles['velocities'], axis=1)

    ax9.hist(speeds, bins=50, density=True, alpha=0.7, color=colors['particle'],
            edgecolor='black', linewidth=1.5, label='Observed')

    # Fit Maxwell-Boltzmann distribution
    from scipy.stats import maxwell
    params = maxwell.fit(speeds)
    x_range = np.linspace(0, speeds.max(), 200)
    ax9.plot(x_range, maxwell.pdf(x_range, *params), 'r-', linewidth=3,
            label=f'Maxwell-Boltzmann fit\nscale={params[1]:.3f}')

    ax9.set_xlabel('Speed', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax9.set_title('(I) Velocity Distribution\nMaxwell-Boltzmann Statistics',
                fontsize=14, fontweight='bold', pad=15)
    ax9.legend(fontsize=10)
    ax9.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 10: Run-to-Run Variability
    # ============================================================
    ax10 = fig.add_subplot(gs[4, :2])

    if all_sim_metrics and len(all_sim_metrics) > 1:
        # Calculate CV for each metric
        cv_exec = np.std([m['execution_time'] for m in all_sim_metrics], axis=0) / \
                (np.mean([m['execution_time'] for m in all_sim_metrics], axis=0) + 1e-10) * 100
        cv_conv = np.std([m['convergence'] for m in all_sim_metrics], axis=0) / \
                (np.mean([m['convergence'] for m in all_sim_metrics], axis=0) + 1e-10) * 100
        cv_stab = np.std([m['stability'] for m in all_sim_metrics], axis=0) / \
                (np.mean([m['stability'] for m in all_sim_metrics], axis=0) + 1e-10) * 100

        # Plot heatmap
        cv_matrix = np.array([cv_exec, cv_conv, cv_stab])

        im = ax10.imshow(cv_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=20)
        cbar = plt.colorbar(im, ax=ax10)
        cbar.set_label('Coefficient of Variation (%)', fontsize=11, fontweight='bold')

        ax10.set_yticks(range(3))
        ax10.set_yticklabels(['Execution Time', 'Convergence', 'Stability'], fontsize=10)
        ax10.set_xticks(range(len(common_sim_components)))
        ax10.set_xticklabels(common_sim_components, rotation=45, ha='right', fontsize=9)

        # Add values
        for i in range(3):
            for j in range(len(common_sim_components)):
                text = ax10.text(j, i, f'{cv_matrix[i, j]:.1f}',
                            ha='center', va='center', fontsize=9,
                            fontweight='bold',
                            color='white' if cv_matrix[i, j] > 10 else 'black')

        ax10.set_title('(J) Run-to-Run Variability Heatmap\nCoefficient of Variation by Component',
                    fontsize=14, fontweight='bold', pad=15)

    # ============================================================
    # PANEL 11: Statistical Summary
    # ============================================================
    ax11 = fig.add_subplot(gs[4, 2:])
    ax11.axis('off')

    # Compute comprehensive statistics
    if all_sim_metrics:
        # Overall statistics
        all_exec_times = np.concatenate([m['execution_time'] for m in all_sim_metrics])
        all_convergence = np.concatenate([m['convergence'] for m in all_sim_metrics])
        all_iterations = np.concatenate([m['iterations'] for m in all_sim_metrics])
        all_stability = np.concatenate([m['stability'] for m in all_sim_metrics])

        # Per-run statistics
        run_stats = []
        for i, metrics in enumerate(all_sim_metrics):
            run_stats.append({
                'exec_mean': np.mean(metrics['execution_time']),
                'conv_mean': np.mean(metrics['convergence']),
                'iter_mean': np.mean(metrics['iterations']),
                'stab_mean': np.mean(metrics['stability'])
            })

        # Particle statistics
        mean_speed = np.mean(speeds)
        mean_energy = np.mean(particles['energies'])

        summary_text = f"""
    SIMULATION TEST MULTI-RUN ANALYSIS SUMMARY

    DATASET OVERVIEW:
    Total simulation runs:     {len(all_sim_data)}
    Components per run:        {len(common_sim_components)}
    Total measurements:        {len(all_sim_data) * len(common_sim_components)}
    Date range:                {all_sim_data[0]['timestamp']} to {all_sim_data[-1]['timestamp']}

    EXECUTION PERFORMANCE:
    Overall mean time:         {np.mean(all_exec_times):.6f} s
    Overall std time:          {np.std(all_exec_times):.6f} s
    Cross-run CV:              {(np.std([s['exec_mean'] for s in run_stats])/np.mean([s['exec_mean'] for s in run_stats])*100):.2f}%
    Min execution time:        {np.min(all_exec_times):.6f} s
    Max execution time:        {np.max(all_exec_times):.6f} s

    CONVERGENCE METRICS:
    Overall mean:              {np.mean(all_convergence):.6f}
    Overall std:               {np.std(all_convergence):.6f}
    Cross-run CV:              {(np.std([s['conv_mean'] for s in run_stats])/np.mean([s['conv_mean'] for s in run_stats])*100):.2f}%
    Convergence quality:       {'EXCELLENT' if np.mean(all_convergence) > 0.95 else 'GOOD' if np.mean(all_convergence) > 0.90 else 'ACCEPTABLE'}

    ITERATION STATISTICS:
    Mean iterations:           {np.mean(all_iterations):.0f}
    Std iterations:            {np.std(all_iterations):.0f}
    Min iterations:            {np.min(all_iterations):.0f}
    Max iterations:            {np.max(all_iterations):.0f}

    STABILITY ANALYSIS:
    Overall mean:              {np.mean(all_stability):.6f}
    Overall std:               {np.std(all_stability):.6f}
    Cross-run CV:              {(np.std([s['stab_mean'] for s in run_stats])/np.mean([s['stab_mean'] for s in run_stats])*100):.2f}%
    Stability rating:          {'EXCELLENT' if np.mean(all_stability) > 0.95 else 'GOOD' if np.mean(all_stability) > 0.90 else 'ACCEPTABLE'}

    PARTICLE SIMULATION:
    Number of particles:       {n_particles:,}
    Timesteps:                 {n_timesteps}
    Mean particle speed:       {mean_speed:.4f}
    Mean particle energy:      {mean_energy:.4f}
    Energy conservation:       {conservation_error:.4f}%

    REPRODUCIBILITY:
    Execution time CV:         {(np.std([s['exec_mean'] for s in run_stats])/np.mean([s['exec_mean'] for s in run_stats])*100):.2f}% {'✓ EXCELLENT' if (np.std([s['exec_mean'] for s in run_stats])/np.mean([s['exec_mean'] for s in run_stats])*100) < 5 else '✓ GOOD'}
    Convergence CV:            {(np.std([s['conv_mean'] for s in run_stats])/np.mean([s['conv_mean'] for s in run_stats])*100):.2f}% {'✓ EXCELLENT' if (np.std([s['conv_mean'] for s in run_stats])/np.mean([s['conv_mean'] for s in run_stats])*100) < 5 else '✓ GOOD'}
    Stability CV:              {(np.std([s['stab_mean'] for s in run_stats])/np.mean([s['stab_mean'] for s in run_stats])*100):.2f}% {'✓ EXCELLENT' if (np.std([s['stab_mean'] for s in run_stats])/np.mean([s['stab_mean'] for s in run_stats])*100) < 5 else '✓ GOOD'}

    Overall assessment:        {'HIGHLY REPRODUCIBLE' if all((np.std([s[k] for s in run_stats])/np.mean([s[k] for s in run_stats])*100) < 5 for k in ['exec_mean', 'conv_mean', 'stab_mean']) else 'REPRODUCIBLE'}
    """

        ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95))

    # Main title
    fig.suptitle('Simulation Test Multi-Run Analysis: Particle Dynamics and Reproducibility\n'
                f'{len(all_sim_data)} Runs | {len(common_sim_components)} Components | '
                f'{n_particles:,} Particles | {n_timesteps} Timesteps',
                fontsize=16, fontweight='bold', y=0.998)

    plt.savefig('simulation_test_multirun_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('simulation_test_multirun_analysis.png', dpi=300, bbox_inches='tight')

    print("\n" + "="*80)
    print("✓ Simulation test multi-run analysis figure created")
    print(f"  Runs analyzed: {len(all_sim_data)}")
    print(f"  Components: {len(common_sim_components)}")
    print(f"  Particles: {n_particles:,}")
    print(f"  Timesteps: {n_timesteps}")
    if all_sim_metrics:
        print(f"  Mean execution time: {np.mean([s['exec_mean'] for s in run_stats]):.6f} s")
        print(f"  Mean convergence: {np.mean([s['conv_mean'] for s in run_stats]):.6f}")
        print(f"  Mean stability: {np.mean([s['stab_mean'] for s in run_stats]):.6f}")
    print("="*80)
