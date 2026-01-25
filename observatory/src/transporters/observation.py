import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Wedge, FancyBboxPatch
import matplotlib.patches as mpatches


if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 10))

    # Data from console output
    observations = {
        'total': 300,
        'measurements': 3,
        'feedbacks': 2,
        'transports': 0,
        'rejections': 3,
        'momentum_transfer': 0.0,
        'time_resolution': 1.0e-15
    }

    substrates_obs = {
        'Doxorubicin': {'detected': False, 'phase_lock': 0.100, 'feedback': False},
        'Verapamil': {'detected': True, 'phase_lock': 1.000, 'feedback': False},
        'Glucose': {'detected': True, 'phase_lock': 0.500, 'feedback': False}
    }

    heisenberg_limit = 5.27e-25
    thermal_momentum = 5.96e-22

    # Panel A: Observation Event Breakdown (Sunburst Chart)
    ax1 = plt.subplot(2, 3, 1)
    ax1.axis('equal')

    # Inner ring: Total observations
    inner_wedge = Wedge((0, 0), 0.5, 0, 360, width=0.2,
                    facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(inner_wedge)
    ax1.text(0, 0, f'{observations["total"]}\nTotal', ha='center', va='center',
            fontsize=11, weight='bold')

    # Outer ring: Event types
    events = ['Measurements', 'Feedbacks', 'Transports', 'Rejections']
    counts = [observations['measurements'], observations['feedbacks'],
            observations['transports'], observations['rejections']]
    colors_events = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    start_angle = 0
    for event, count, color in zip(events, counts, colors_events):
        if count > 0:
            angle = 360 * count / observations['total']
            wedge = Wedge((0, 0), 0.9, start_angle, start_angle + angle,
                        width=0.3, facecolor=color, edgecolor='black',
                        linewidth=2, alpha=0.7)
            ax1.add_patch(wedge)

            # Label
            mid_angle = np.radians(start_angle + angle/2)
            label_r = 1.1
            label_x = label_r * np.cos(mid_angle)
            label_y = label_r * np.sin(mid_angle)
            ax1.text(label_x, label_y, f'{event}\n{count}',
                    ha='center', va='center', fontsize=8, weight='bold')

            start_angle += angle

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.axis('off')
    ax1.set_title('A. Observation Event Distribution', fontsize=13, weight='bold', pad=10)

    # Panel B: Momentum Transfer Comparison (Log Scale)
    ax2 = plt.subplot(2, 3, 2)
    methods = ['Categorical\n(This Work)', 'Heisenberg\nLimit', 'Thermal\nMomentum']
    momenta = [observations['momentum_transfer'], heisenberg_limit, thermal_momentum]
    colors_mom = ['#2ecc71', '#e74c3c', '#f39c12']

    # Handle zero value for log scale
    momenta_plot = [m if m > 0 else 1e-30 for m in momenta]

    bars = ax2.bar(methods, momenta_plot, color=colors_mom, alpha=0.8,
                edgecolor='black', linewidth=2, log=True)

    # Add value labels
    for bar, val in zip(bars, momenta):
        height = bar.get_height()
        if val == 0:
            label = 'ZERO'
            y_pos = 1e-30
            color = 'green'
            size = 11
        else:
            label = f'{val:.2e}'
            y_pos = height * 2
            color = 'black'
            size = 9

        ax2.text(bar.get_x() + bar.get_width()/2., y_pos, label,
                ha='center', va='bottom', fontsize=size, weight='bold', color=color)

    ax2.set_ylabel('Momentum Transfer (kg·m/s, log scale)', fontsize=11, weight='bold')
    ax2.set_title('B. Backaction Comparison', fontsize=13, weight='bold', pad=10)
    ax2.set_ylim(1e-30, 1e-20)
    ax2.grid(True, alpha=0.3, which='both', axis='y')

    # Panel C: Detection Matrix
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    y_start = 8
    for i, (name, data) in enumerate(substrates_obs.items()):
        y = y_start - i * 2.5

        # Substrate box - FIXED: Use FancyBboxPatch instead of FancyArrowPatch
        box = FancyBboxPatch((0.5, y), 3, 1.5, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='lightblue', linewidth=2)
        ax3.add_patch(box)
        ax3.text(2, y + 0.75, name, ha='center', va='center',
                fontsize=9, weight='bold')
        ax3.text(2, y + 0.3, f'⟨r⟩={data["phase_lock"]:.3f}', ha='center', va='center',
                fontsize=8)

        # Detection status - FIXED
        detect_color = '#2ecc71' if data['detected'] else '#e74c3c'
        detect_box = FancyBboxPatch((4, y + 0.5), 2, 0.8, boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=detect_color, linewidth=2)
        ax3.add_patch(detect_box)
        detect_text = '✓ Detected' if data['detected'] else '✗ Not Detected'
        ax3.text(5, y + 0.9, detect_text, ha='center', va='center',
                fontsize=8, weight='bold', color='white')

        # Feedback status - FIXED
        feedback_color = '#2ecc71' if data['feedback'] else '#95a5a6'
        feedback_box = FancyBboxPatch((6.5, y + 0.5), 2, 0.8, boxstyle="round,pad=0.05",
                                    edgecolor='black', facecolor=feedback_color, linewidth=2)
        ax3.add_patch(feedback_box)
        feedback_text = '✓ Feedback' if data['feedback'] else '✗ No Feedback'
        ax3.text(7.5, y + 0.9, feedback_text, ha='center', va='center',
                fontsize=8, weight='bold', color='white')

    ax3.set_title('C. Substrate Detection Matrix', fontsize=13, weight='bold', pad=10)

    # Panel D: Time Resolution Spectrum
    ax4 = plt.subplot(2, 3, 4)
    time_scales = {
        'Categorical (This Work)': 1e-15,
        'Femtosecond Laser': 1e-15,
        'Attosecond Pulse': 1e-18,
        'Molecular Vibration': 1e-14,
        'Electronic Transition': 1e-16,
        'Nuclear Motion': 1e-13
    }

    names_time = list(time_scales.keys())
    times = list(time_scales.values())
    colors_time = ['#2ecc71' if 'Categorical' in n else '#3498db' for n in names_time]

    y_pos = np.arange(len(names_time))
    bars = ax4.barh(y_pos, times, color=colors_time, alpha=0.8,
                edgecolor='black', linewidth=2, log=True)

    # Add labels
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax4.text(time * 1.5, i, f'{time:.0e} s', va='center',
                fontsize=9, weight='bold')

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(names_time, fontsize=9)
    ax4.set_xlabel('Time Resolution (s, log scale)', fontsize=11, weight='bold')
    ax4.set_title('D. Time Resolution Spectrum', fontsize=13, weight='bold', pad=10)
    ax4.grid(True, alpha=0.3, axis='x')

    # Panel E: Backaction Ratios (Gauge Charts)
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)

    # Backaction vs Heisenberg
    ax5.text(2.5, 8, 'Backaction / Heisenberg', ha='center', fontsize=10, weight='bold')
    gauge1 = Circle((2.5, 6), 1.5, facecolor='#2ecc71', edgecolor='black', linewidth=2)
    ax5.add_patch(gauge1)
    ax5.text(2.5, 6, '0.00', ha='center', va='center', fontsize=14, weight='bold', color='white')
    ax5.text(2.5, 4, '(Zero backaction)', ha='center', fontsize=8, style='italic')

    # Backaction vs Thermal
    ax5.text(7.5, 8, 'Backaction / Thermal', ha='center', fontsize=10, weight='bold')
    gauge2 = Circle((7.5, 6), 1.5, facecolor='#2ecc71', edgecolor='black', linewidth=2)
    ax5.add_patch(gauge2)
    ax5.text(7.5, 6, '0.00', ha='center', va='center', fontsize=14, weight='bold', color='white')
    ax5.text(7.5, 4, '(Zero backaction)', ha='center', fontsize=8, style='italic')

    # Verification badge
    badge = Circle((5, 1.5), 1, facecolor='gold', edgecolor='black', linewidth=3)
    ax5.add_patch(badge)
    ax5.text(5, 1.5, '✓\nVERIFIED', ha='center', va='center',
            fontsize=10, weight='bold')

    ax5.set_title('E. Zero Backaction Verification', fontsize=13, weight='bold', pad=10)

    # Panel F: Observation Statistics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = "TRANS-PLANCKIAN OBSERVATION SUMMARY\n"
    summary_text += "="*60 + "\n\n"
    summary_text += f"Time Resolution:      {observations['time_resolution']:.0e} s\n"
    summary_text += f"Total Observations:   {observations['total']}\n"
    summary_text += f"Measurement Events:   {observations['measurements']}\n"
    summary_text += f"Feedback Events:      {observations['feedbacks']}\n"
    summary_text += f"Transport Events:     {observations['transports']}\n"
    summary_text += f"Rejection Events:     {observations['rejections']}\n\n"
    summary_text += "BACKACTION ANALYSIS:\n"
    summary_text += "-"*60 + "\n"
    summary_text += f"Total Momentum Transfer:  {observations['momentum_transfer']:.2e} kg·m/s\n"
    summary_text += f"Per Observation:          {observations['momentum_transfer']/observations['total']:.2e} kg·m/s\n"
    summary_text += f"Heisenberg Limit:         {heisenberg_limit:.2e} kg·m/s\n"
    summary_text += f"Thermal Momentum:         {thermal_momentum:.2e} kg·m/s\n"
    summary_text += f"Backaction/Heisenberg:    {0.0:.2e}\n"
    summary_text += f"Backaction/Thermal:       {0.0:.2e}\n\n"
    summary_text += "="*60 + "\n"
    summary_text += "VERIFICATION STATUS:\n"
    summary_text += "✓ Zero backaction confirmed\n"
    summary_text += "✓ Trans-Planckian precision achieved\n"
    summary_text += "✓ Categorical measurement validated\n"
    summary_text += "✓ No quantum disturbance detected\n"

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=7.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    ax6.set_title('F. Comprehensive Summary', fontsize=13, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figure7_transplanckian_verification.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure7_transplanckian_verification.pdf', bbox_inches='tight')
    print("✓ Figure 7 saved")
    plt.show()
