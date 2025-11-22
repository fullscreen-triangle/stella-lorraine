import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import seaborn as sns
from datetime import datetime


if __name__ == "__main__":
    # Load all navigation test files
    navigation_files = [
        'public/navigation_test_20251011_050633.json',
        'public/navigation_test_20251105_114617.json',
        'public/navigation_test_20251105_114703.json',
        'public/navigation_test_20251105_115638.json',
        'public/navigation_test_20251105_120856.json',
        'public/navigation_test_20251105_123223.json'
    ]

    print("="*80)
    print("NAVIGATION TEST ANALYSIS - MULTI-RUN COMPARISON")
    print("="*80)

    # Load all data
    all_data = []
    for filename in navigation_files:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                all_data.append(data)
                print(f"✓ Loaded: {filename}")
                print(f"  Timestamp: {data['timestamp']}")
                print(f"  Components: {len(data.get('components_tested', []))}")
        except Exception as e:
            print(f"✗ Failed to load {filename}: {e}")

    print(f"\nTotal runs loaded: {len(all_data)}")
    print("="*80)

    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(5, 4, figure=fig, hspace=0.45, wspace=0.4)

    # Color scheme
    colors = {
        'run1': '#3498db',
        'run2': '#e74c3c',
        'run3': '#2ecc71',
        'run4': '#9b59b6',
        'run5': '#f39c12',
        'run6': '#1abc9c',
        'mean': '#34495e',
        'std': '#95a5a6'
    }

    run_colors = [colors[f'run{i+1}'] for i in range(len(all_data))]

    # Extract component data from all runs
    def extract_component_metrics(data):
        """Extract metrics from components_tested"""
        components = data.get('components_tested', [])
        metrics = {
            'names': [],
            'status': [],
            'execution_time': [],
            'memory_usage': [],
            'accuracy': [],
            'precision': []
        }

        for comp in components:
            metrics['names'].append(comp.get('name', 'Unknown'))
            metrics['status'].append(comp.get('status', 'unknown'))
            metrics['execution_time'].append(comp.get('execution_time', 0))
            metrics['memory_usage'].append(comp.get('memory_usage', 0))
            metrics['accuracy'].append(comp.get('accuracy', 0))
            metrics['precision'].append(comp.get('precision', 0))

        return metrics

    # Extract metrics for all runs
    all_metrics = [extract_component_metrics(data) for data in all_data]

    # Get common component names
    common_components = all_metrics[0]['names'] if all_metrics else []

    # ============================================================
    # PANEL 1: Execution Time Comparison Across Runs
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :2])

    x_pos = np.arange(len(common_components))
    width = 0.12

    for i, (metrics, color) in enumerate(zip(all_metrics, run_colors)):
        exec_times = metrics['execution_time']
        offset = (i - len(all_metrics)/2) * width
        ax1.bar(x_pos + offset, exec_times, width,
            label=f'Run {i+1} ({all_data[i]["timestamp"]})',
            color=color, alpha=0.8, edgecolor='black', linewidth=1)

    # Add mean line
    if all_metrics:
        mean_times = np.mean([m['execution_time'] for m in all_metrics], axis=0)
        ax1.plot(x_pos, mean_times, 'k--', linewidth=3, marker='o',
                markersize=8, label='Mean across runs')

    ax1.set_xlabel('Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (s)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Execution Time Comparison Across All Runs\nComponent-Level Performance',
                fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(common_components, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=8, loc='upper left', ncol=2)
    ax1.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 2: Memory Usage Comparison
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 2:])

    for i, (metrics, color) in enumerate(zip(all_metrics, run_colors)):
        mem_usage = metrics['memory_usage']
        offset = (i - len(all_metrics)/2) * width
        ax2.bar(x_pos + offset, mem_usage, width,
            label=f'Run {i+1}',
            color=color, alpha=0.8, edgecolor='black', linewidth=1)

    # Add mean line
    if all_metrics:
        mean_mem = np.mean([m['memory_usage'] for m in all_metrics], axis=0)
        ax2.plot(x_pos, mean_mem, 'k--', linewidth=3, marker='s',
                markersize=8, label='Mean')

    ax2.set_xlabel('Component', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Memory Usage Comparison\nResource Consumption Analysis',
                fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(common_components, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=8, loc='upper left', ncol=2)
    ax2.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 3: Accuracy Comparison
    # ============================================================
    ax3 = fig.add_subplot(gs[1, :2])

    for i, (metrics, color) in enumerate(zip(all_metrics, run_colors)):
        accuracy = metrics['accuracy']
        ax3.plot(x_pos, accuracy, 'o-', linewidth=2, markersize=8,
                color=color, label=f'Run {i+1}', alpha=0.7)

    # Add mean and std bands
    if all_metrics:
        mean_acc = np.mean([m['accuracy'] for m in all_metrics], axis=0)
        std_acc = np.std([m['accuracy'] for m in all_metrics], axis=0)

        ax3.plot(x_pos, mean_acc, 'k-', linewidth=4, marker='D',
                markersize=10, label='Mean', zorder=10)
        ax3.fill_between(x_pos, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.3, color='gray', label='±1σ')

    ax3.set_xlabel('Component', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Accuracy Across Runs\nConsistency Analysis',
                fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(common_components, rotation=45, ha='right', fontsize=9)
    ax3.legend(fontsize=8, loc='lower right', ncol=2)
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.set_ylim(0, 105)

    # ============================================================
    # PANEL 4: Precision Comparison
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 2:])

    for i, (metrics, color) in enumerate(zip(all_metrics, run_colors)):
        precision = metrics['precision']
        ax4.plot(x_pos, precision, 's-', linewidth=2, markersize=8,
                color=color, label=f'Run {i+1}', alpha=0.7)

    # Add mean and std bands
    if all_metrics:
        mean_prec = np.mean([m['precision'] for m in all_metrics], axis=0)
        std_prec = np.std([m['precision'] for m in all_metrics], axis=0)

        ax4.plot(x_pos, mean_prec, 'k-', linewidth=4, marker='D',
                markersize=10, label='Mean', zorder=10)
        ax4.fill_between(x_pos, mean_prec - std_prec, mean_prec + std_prec,
                        alpha=0.3, color='gray', label='±1σ')

    ax4.set_xlabel('Component', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
    ax4.set_title('(D) Precision Across Runs\nReproducibility Assessment',
                fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(common_components, rotation=45, ha='right', fontsize=9)
    ax4.legend(fontsize=8, loc='lower right', ncol=2)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_ylim(0, 105)

    # ============================================================
    # PANEL 5: Run-to-Run Variability (CV)
    # ============================================================
    ax5 = fig.add_subplot(gs[2, 0])

    if all_metrics and len(all_metrics) > 1:
        # Calculate coefficient of variation for each metric
        cv_exec = np.std([m['execution_time'] for m in all_metrics], axis=0) / \
                (np.mean([m['execution_time'] for m in all_metrics], axis=0) + 1e-10) * 100
        cv_mem = np.std([m['memory_usage'] for m in all_metrics], axis=0) / \
                (np.mean([m['memory_usage'] for m in all_metrics], axis=0) + 1e-10) * 100
        cv_acc = np.std([m['accuracy'] for m in all_metrics], axis=0) / \
                (np.mean([m['accuracy'] for m in all_metrics], axis=0) + 1e-10) * 100
        cv_prec = np.std([m['precision'] for m in all_metrics], axis=0) / \
                (np.mean([m['precision'] for m in all_metrics], axis=0) + 1e-10) * 100

        # Average CV across components
        metrics_cv = {
            'Execution Time': np.mean(cv_exec),
            'Memory Usage': np.mean(cv_mem),
            'Accuracy': np.mean(cv_acc),
            'Precision': np.mean(cv_prec)
        }

        bars = ax5.bar(metrics_cv.keys(), metrics_cv.values(),
                    color=[colors['run1'], colors['run2'], colors['run3'], colors['run4']],
                    alpha=0.8, edgecolor='black', linewidth=2)

        # Color bars by CV level
        for bar, cv in zip(bars, metrics_cv.values()):
            if cv < 5:
                bar.set_color(colors['run3'])  # Green - excellent
            elif cv < 10:
                bar.set_color(colors['run5'])  # Orange - good
            else:
                bar.set_color(colors['run2'])  # Red - high variability

        # Add value labels
        for i, (name, cv) in enumerate(metrics_cv.items()):
            ax5.text(i, cv + 0.5, f'{cv:.2f}%', ha='center', fontsize=10,
                    fontweight='bold')

        # Add reference lines
        ax5.axhline(5, color='green', linestyle='--', linewidth=2, alpha=0.5,
                label='Excellent (<5%)')
        ax5.axhline(10, color='orange', linestyle='--', linewidth=2, alpha=0.5,
                label='Good (<10%)')

    ax5.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax5.set_title('(E) Run-to-Run Variability\nCoefficient of Variation',
                fontsize=14, fontweight='bold', pad=15)
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 6: Success Rate Across Runs
    # ============================================================
    ax6 = fig.add_subplot(gs[2, 1])

    success_rates = []
    for i, metrics in enumerate(all_metrics):
        n_success = sum(1 for s in metrics['status'] if s.lower() == 'pass')
        n_total = len(metrics['status'])
        success_rate = (n_success / n_total * 100) if n_total > 0 else 0
        success_rates.append(success_rate)

    bars = ax6.bar(range(1, len(success_rates) + 1), success_rates,
                color=run_colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Add value labels
    for i, rate in enumerate(success_rates):
        ax6.text(i + 1, rate + 1, f'{rate:.1f}%', ha='center',
                fontsize=10, fontweight='bold')

    # Add mean line
    mean_success = np.mean(success_rates)
    ax6.axhline(mean_success, color='black', linestyle='--', linewidth=3,
            label=f'Mean: {mean_success:.1f}%')

    ax6.set_xlabel('Run Number', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax6.set_title('(F) Success Rate by Run\nComponent Pass/Fail',
                fontsize=14, fontweight='bold', pad=15)
    ax6.set_xticks(range(1, len(success_rates) + 1))
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3, linestyle='--', axis='y')
    ax6.set_ylim(0, 105)

    # ============================================================
    # PANEL 7: Correlation Matrix (Mean Metrics)
    # ============================================================
    ax7 = fig.add_subplot(gs[2, 2:])

    if all_metrics:
        # Calculate mean metrics across runs
        mean_exec = np.mean([m['execution_time'] for m in all_metrics], axis=0)
        mean_mem = np.mean([m['memory_usage'] for m in all_metrics], axis=0)
        mean_acc = np.mean([m['accuracy'] for m in all_metrics], axis=0)
        mean_prec = np.mean([m['precision'] for m in all_metrics], axis=0)

        # Create correlation matrix
        data_matrix = np.array([mean_exec, mean_mem, mean_acc, mean_prec])
        corr_matrix = np.corrcoef(data_matrix)

        # Plot heatmap
        im = ax7.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax7)
        cbar.set_label('Correlation', fontsize=11, fontweight='bold')

        # Add labels
        labels = ['Exec Time', 'Memory', 'Accuracy', 'Precision']
        ax7.set_xticks(range(4))
        ax7.set_yticks(range(4))
        ax7.set_xticklabels(labels, fontsize=10)
        ax7.set_yticklabels(labels, fontsize=10)

        # Add correlation values
        for i in range(4):
            for j in range(4):
                text = ax7.text(j, i, f'{corr_matrix[i, j]:.2f}',
                            ha='center', va='center', fontsize=11,
                            fontweight='bold',
                            color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    ax7.set_title('(G) Metric Correlation Matrix\nCross-Metric Relationships',
                fontsize=14, fontweight='bold', pad=15)

    # ============================================================
    # PANEL 8: Temporal Evolution (if timestamps available)
    # ============================================================
    ax8 = fig.add_subplot(gs[3, :2])

    # Extract timestamps and convert to datetime
    timestamps = []
    for data in all_data:
        ts_str = data['timestamp']
        try:
            # Parse timestamp (format: YYYYMMDD_HHMMSS)
            dt = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
            timestamps.append(dt)
        except:
            timestamps.append(None)

    if all(t is not None for t in timestamps):
        # Calculate time differences from first run
        time_diffs = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]  # hours

        # Plot metrics evolution over time
        mean_exec_per_run = [np.mean(m['execution_time']) for m in all_metrics]
        mean_mem_per_run = [np.mean(m['memory_usage']) for m in all_metrics]
        mean_acc_per_run = [np.mean(m['accuracy']) for m in all_metrics]
        mean_prec_per_run = [np.mean(m['precision']) for m in all_metrics]

        ax8_twin1 = ax8.twinx()
        ax8_twin2 = ax8.twinx()
        ax8_twin2.spines['right'].set_position(('outward', 60))

        l1 = ax8.plot(time_diffs, mean_exec_per_run, 'o-', linewidth=3,
                    markersize=10, color=colors['run1'], label='Exec Time (s)')
        l2 = ax8_twin1.plot(time_diffs, mean_mem_per_run, 's-', linewidth=3,
                        markersize=10, color=colors['run2'], label='Memory (MB)')
        l3 = ax8_twin2.plot(time_diffs, mean_acc_per_run, '^-', linewidth=3,
                        markersize=10, color=colors['run3'], label='Accuracy (%)')

        ax8.set_xlabel('Time Since First Run (hours)', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Execution Time (s)', fontsize=11, fontweight='bold', color=colors['run1'])
        ax8_twin1.set_ylabel('Memory Usage (MB)', fontsize=11, fontweight='bold', color=colors['run2'])
        ax8_twin2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color=colors['run3'])

        ax8.tick_params(axis='y', labelcolor=colors['run1'])
        ax8_twin1.tick_params(axis='y', labelcolor=colors['run2'])
        ax8_twin2.tick_params(axis='y', labelcolor=colors['run3'])

        # Combine legends
        lns = l1 + l2 + l3
        labs = [l.get_label() for l in lns]
        ax8.legend(lns, labs, fontsize=10, loc='upper left')

        ax8.set_title('(H) Temporal Evolution of Metrics\nPerformance Over Time',
                    fontsize=14, fontweight='bold', pad=15)
        ax8.grid(alpha=0.3, linestyle='--')
    else:
        ax8.text(0.5, 0.5, 'Timestamp parsing failed\nCannot plot temporal evolution',
                ha='center', va='center', transform=ax8.transAxes,
                fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        ax8.set_title('(H) Temporal Evolution\n(Data unavailable)',
                    fontsize=14, fontweight='bold', pad=15)

    # ============================================================
    # PANEL 9: Component-wise Stability
    # ============================================================
    ax9 = fig.add_subplot(gs[3, 2:])

    if all_metrics and len(all_metrics) > 1:
        # Calculate stability score for each component
        # Stability = 1 / (1 + CV)
        stability_scores = []

        for i in range(len(common_components)):
            exec_vals = [m['execution_time'][i] for m in all_metrics]
            acc_vals = [m['accuracy'][i] for m in all_metrics]
            prec_vals = [m['precision'][i] for m in all_metrics]

            # Calculate CV for each metric
            cv_exec_comp = np.std(exec_vals) / (np.mean(exec_vals) + 1e-10)
            cv_acc_comp = np.std(acc_vals) / (np.mean(acc_vals) + 1e-10)
            cv_prec_comp = np.std(prec_vals) / (np.mean(prec_vals) + 1e-10)

            # Overall stability (inverse of mean CV)
            mean_cv = (cv_exec_comp + cv_acc_comp + cv_prec_comp) / 3
            stability = 1 / (1 + mean_cv)
            stability_scores.append(stability * 100)  # Convert to percentage

        bars = ax9.barh(common_components, stability_scores, color=colors['run4'],
                    alpha=0.8, edgecolor='black', linewidth=1.5)

        # Color bars by stability
        for bar, score in zip(bars, stability_scores):
            if score > 90:
                bar.set_color(colors['run3'])  # Green - very stable
            elif score > 80:
                bar.set_color(colors['run5'])  # Orange - stable
            else:
                bar.set_color(colors['run2'])  # Red - unstable

        # Add value labels
        for i, score in enumerate(stability_scores):
            ax9.text(score + 1, i, f'{score:.1f}%', va='center',
                    fontsize=10, fontweight='bold')

        # Add reference lines
        ax9.axvline(90, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax9.axvline(80, color='orange', linestyle='--', linewidth=2, alpha=0.5)

    ax9.set_xlabel('Stability Score (%)', fontsize=12, fontweight='bold')
    ax9.set_title('(I) Component-wise Stability\nConsistency Across Runs',
                fontsize=14, fontweight='bold', pad=15)
    ax9.grid(alpha=0.3, linestyle='--', axis='x')
    ax9.set_xlim(0, 105)

    # ============================================================
    # PANEL 10: Statistical Summary
    # ============================================================
    ax10 = fig.add_subplot(gs[4, :])
    ax10.axis('off')

    # Compute comprehensive statistics
    if all_metrics:
        # Overall statistics
        all_exec_times = np.concatenate([m['execution_time'] for m in all_metrics])
        all_mem_usage = np.concatenate([m['memory_usage'] for m in all_metrics])
        all_accuracy = np.concatenate([m['accuracy'] for m in all_metrics])
        all_precision = np.concatenate([m['precision'] for m in all_metrics])

        # Per-run statistics
        run_stats = []
        for i, metrics in enumerate(all_metrics):
            run_stats.append({
                'exec_mean': np.mean(metrics['execution_time']),
                'exec_std': np.std(metrics['execution_time']),
                'mem_mean': np.mean(metrics['memory_usage']),
                'mem_std': np.std(metrics['memory_usage']),
                'acc_mean': np.mean(metrics['accuracy']),
                'prec_mean': np.mean(metrics['precision']),
                'success_rate': sum(1 for s in metrics['status'] if s.lower() == 'pass') / len(metrics['status']) * 100
            })

        # Cross-run statistics
        exec_means = [s['exec_mean'] for s in run_stats]
        mem_means = [s['mem_mean'] for s in run_stats]
        acc_means = [s['acc_mean'] for s in run_stats]
        prec_means = [s['prec_mean'] for s in run_stats]
        success_rates_all = [s['success_rate'] for s in run_stats]

        summary_text = f"""
    NAVIGATION TEST MULTI-RUN ANALYSIS SUMMARY

    DATASET OVERVIEW:
    Total runs analyzed:       {len(all_data)}
    Components per run:        {len(common_components)}
    Total measurements:        {len(all_data) * len(common_components)}
    Date range:                {all_data[0]['timestamp']} to {all_data[-1]['timestamp']}

    EXECUTION TIME STATISTICS:
    Overall mean:              {np.mean(all_exec_times):.6f} s
    Overall std:               {np.std(all_exec_times):.6f} s
    Cross-run mean:            {np.mean(exec_means):.6f} s
    Cross-run std:             {np.std(exec_means):.6f} s
    Cross-run CV:              {(np.std(exec_means)/np.mean(exec_means)*100):.2f}%
    Min (any run):             {np.min(all_exec_times):.6f} s
    Max (any run):             {np.max(all_exec_times):.6f} s

    MEMORY USAGE STATISTICS:
    Overall mean:              {np.mean(all_mem_usage):.2f} MB
    Overall std:               {np.std(all_mem_usage):.2f} MB
    Cross-run mean:            {np.mean(mem_means):.2f} MB
    Cross-run std:             {np.std(mem_means):.2f} MB
    Cross-run CV:              {(np.std(mem_means)/np.mean(mem_means)*100):.2f}%
    Min (any run):             {np.min(all_mem_usage):.2f} MB
    Max (any run):             {np.max(all_mem_usage):.2f} MB

    ACCURACY STATISTICS:
    Overall mean:              {np.mean(all_accuracy):.2f}%
    Overall std:               {np.std(all_accuracy):.2f}%
    Cross-run mean:            {np.mean(acc_means):.2f}%
    Cross-run std:             {np.std(acc_means):.2f}%
    Cross-run CV:              {(np.std(acc_means)/np.mean(acc_means)*100):.2f}%
    Min (any run):             {np.min(all_accuracy):.2f}%
    Max (any run):             {np.max(all_accuracy):.2f}%

    PRECISION STATISTICS:
    Overall mean:              {np.mean(all_precision):.2f}%
    Overall std:               {np.std(all_precision):.2f}%
    Cross-run mean:            {np.mean(prec_means):.2f}%
    Cross-run std:             {np.std(prec_means):.2f}%
    Cross-run CV:              {(np.std(prec_means)/np.mean(prec_means)*100):.2f}%
    Min (any run):             {np.min(all_precision):.2f}%
    Max (any run):             {np.max(all_precision):.2f}%

    SUCCESS RATE:
    Mean across runs:          {np.mean(success_rates_all):.2f}%
    Std across runs:           {np.std(success_rates_all):.2f}%
    Min success rate:          {np.min(success_rates_all):.2f}%
    Max success rate:          {np.max(success_rates_all):.2f}%
    Consistency:               {'EXCELLENT' if np.std(success_rates_all) < 5 else 'GOOD' if np.std(success_rates_all) < 10 else 'VARIABLE'}

    REPRODUCIBILITY ASSESSMENT:
    Execution time CV:         {(np.std(exec_means)/np.mean(exec_means)*100):.2f}% {'✓ EXCELLENT' if (np.std(exec_means)/np.mean(exec_means)*100) < 5 else '✓ GOOD' if (np.std(exec_means)/np.mean(exec_means)*100) < 10 else '⚠ VARIABLE'}
    Memory usage CV:           {(np.std(mem_means)/np.mean(mem_means)*100):.2f}% {'✓ EXCELLENT' if (np.std(mem_means)/np.mean(mem_means)*100) < 5 else '✓ GOOD' if (np.std(mem_means)/np.mean(mem_means)*100) < 10 else '⚠ VARIABLE'}
    Accuracy CV:               {(np.std(acc_means)/np.mean(acc_means)*100):.2f}% {'✓ EXCELLENT' if (np.std(acc_means)/np.mean(acc_means)*100) < 5 else '✓ GOOD' if (np.std(acc_means)/np.mean(acc_means)*100) < 10 else '⚠ VARIABLE'}
    Precision CV:              {(np.std(prec_means)/np.mean(prec_means)*100):.2f}% {'✓ EXCELLENT' if (np.std(prec_means)/np.mean(prec_means)*100) < 5 else '✓ GOOD' if (np.std(prec_means)/np.mean(prec_means)*100) < 10 else '⚠ VARIABLE'}

    Overall reproducibility:   {'EXCELLENT' if all((np.std(m)/np.mean(m)*100) < 5 for m in [exec_means, mem_means, acc_means, prec_means]) else 'GOOD' if all((np.std(m)/np.mean(m)*100) < 10 for m in [exec_means, mem_means, acc_means, prec_means]) else 'ACCEPTABLE'}

    BEST PERFORMING RUN:
    Run number:                {np.argmax(success_rates_all) + 1}
    Timestamp:                 {all_data[np.argmax(success_rates_all)]['timestamp']}
    Success rate:              {np.max(success_rates_all):.2f}%
    Mean execution time:       {exec_means[np.argmax(success_rates_all)]:.6f} s
    Mean accuracy:             {acc_means[np.argmax(success_rates_all)]:.2f}%
    """

        ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95))

    # Main title
    fig.suptitle('Navigation Test Multi-Run Analysis: Reproducibility and Consistency Assessment\n'
                f'{len(all_data)} Runs | {len(common_components)} Components | '
                f'Date Range: {all_data[0]["timestamp"]} to {all_data[-1]["timestamp"]}',
                fontsize=16, fontweight='bold', y=0.998)

    plt.savefig('navigation_test_multirun_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('navigation_test_multirun_analysis.png', dpi=300, bbox_inches='tight')

    print("\n" + "="*80)
    print("✓ Navigation test multi-run analysis figure created")
    print(f"  Runs analyzed: {len(all_data)}")
    print(f"  Components: {len(common_components)}")
    if all_metrics:
        print(f"  Mean execution time: {np.mean(exec_means):.6f} s")
        print(f"  Mean accuracy: {np.mean(acc_means):.2f}%")
        print(f"  Mean success rate: {np.mean(success_rates_all):.2f}%")
    print("="*80)
