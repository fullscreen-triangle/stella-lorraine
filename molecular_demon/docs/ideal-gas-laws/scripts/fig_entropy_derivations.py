"""
Figure: Three Entropy Derivations
Shows the derivation of entropy from three equivalent perspectives:
1. Categorical: S = k_B M ln n
2. Oscillatory: S = k_B sum ln(A_i/A_0)
3. Partition: S = k_B sum ln(1/s_a)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import json
import os

# Constants
k_B = 1.380649e-23

# Style settings
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

colors = {
    'categorical': '#2ecc71',
    'oscillatory': '#3498db',
    'partition': '#e74c3c',
    'box': '#ecf0f1',
}

def create_figure():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # =========================================
    # Panel A: Categorical Entropy Derivation
    # =========================================
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'CATEGORICAL ENTROPY', fontsize=14, fontweight='bold',
            ha='center', color=colors['categorical'])
    
    # Draw M categories as boxes
    M = 5
    n = 4  # states per category
    box_width = 1.5
    box_height = 1.0
    
    for i in range(M):
        x = 1 + i * 1.8
        y = 6
        # Category box
        rect = FancyBboxPatch((x, y), box_width, box_height, 
                               boxstyle="round,pad=0.05",
                               facecolor=colors['categorical'], alpha=0.3,
                               edgecolor=colors['categorical'], linewidth=2)
        ax.add_patch(rect)
        ax.text(x + box_width/2, y + box_height/2, f'C{i+1}', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Show n states within each category
        for j in range(n):
            dot_x = x + 0.3 + j * 0.35
            dot_y = y - 0.5
            ax.plot(dot_x, dot_y, 'o', markersize=6, color=colors['categorical'])
    
    ax.text(5, 5, f'M = {M} categories', ha='center', fontsize=11)
    ax.text(5, 4.3, f'n = {n} states each', ha='center', fontsize=11)
    
    # Derivation
    ax.text(5, 3.2, r'Total configurations: $W = n^M$', ha='center', fontsize=11)
    ax.text(5, 2.4, r'$S = k_B \ln W = k_B \ln(n^M)$', ha='center', fontsize=12)
    
    # Final result in box
    result_box = FancyBboxPatch((1.5, 0.5), 7, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(result_box)
    ax.text(5, 1.1, r'$S_{cat} = k_B M \ln n$', ha='center', fontsize=14, fontweight='bold')
    
    # =========================================
    # Panel B: Oscillatory Entropy Derivation
    # =========================================
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'OSCILLATORY ENTROPY', fontsize=14, fontweight='bold',
            ha='center', color=colors['oscillatory'])
    
    # Draw oscillators with different amplitudes
    N_osc = 4
    amplitudes = [0.3, 0.6, 0.4, 0.8]  # Different amplitudes
    A_0 = 0.2  # Reference amplitude
    
    for i, A in enumerate(amplitudes):
        x_center = 1.5 + i * 2
        y_center = 7
        
        # Draw oscillation as ellipse/spring
        theta = np.linspace(0, 2*np.pi, 100)
        x_osc = x_center + A * np.cos(theta)
        y_osc = y_center + 0.3 * np.sin(theta)
        ax.plot(x_osc, y_osc, '-', color=colors['oscillatory'], linewidth=2)
        ax.plot(x_center, y_center, 'ko', markersize=8)
        
        # Amplitude arrow
        ax.annotate('', xy=(x_center + A, y_center), xytext=(x_center, y_center),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(x_center + A/2, y_center - 0.4, f'$A_{i+1}$', ha='center', fontsize=9)
    
    ax.text(5, 5.5, 'Phase space volume:', ha='center', fontsize=11)
    ax.text(5, 4.8, r'$\Gamma_i = \pi m \omega A_i^2$', ha='center', fontsize=11)
    
    ax.text(5, 3.8, 'Ratio to ground state:', ha='center', fontsize=11)
    ax.text(5, 3.1, r'$\Gamma_i / \Gamma_0 = (A_i/A_0)^2$', ha='center', fontsize=11)
    
    ax.text(5, 2.2, r'$S = k_B \sum_i \ln(\Gamma_i/\Gamma_0)$', ha='center', fontsize=12)
    
    # Final result
    result_box = FancyBboxPatch((0.8, 0.5), 8.4, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightcyan', edgecolor='black', linewidth=2)
    ax.add_patch(result_box)
    ax.text(5, 1.1, r'$S_{osc} = k_B \sum_i \ln(A_i/A_0)^2$', 
            ha='center', fontsize=14, fontweight='bold')
    
    # =========================================
    # Panel C: Partition Entropy Derivation
    # =========================================
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'PARTITION ENTROPY', fontsize=14, fontweight='bold',
            ha='center', color=colors['partition'])
    
    # Draw apertures with different selectivities
    N_ap = 4
    selectivities = [0.5, 0.25, 0.4, 0.1]  # Different selectivities
    
    for i, s in enumerate(selectivities):
        x = 1.5 + i * 2
        y = 7
        
        # Aperture as funnel shape
        aperture_width = 0.8 * (1 - s)  # Narrower = lower selectivity
        
        # Draw funnel
        ax.fill([x - 0.5, x + 0.5, x + aperture_width/2, x - aperture_width/2],
                [y + 0.8, y + 0.8, y - 0.8, y - 0.8],
                color=colors['partition'], alpha=0.3, edgecolor=colors['partition'], linewidth=2)
        
        # Arrows showing particles going through
        ax.annotate('', xy=(x, y - 1.2), xytext=(x, y + 1.2),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))
        
        ax.text(x, y - 1.6, f'$s_{i+1}={s}$', ha='center', fontsize=9)
    
    ax.text(5, 5.5, 'Selectivity:', ha='center', fontsize=11)
    ax.text(5, 4.8, r'$s_a = 1/n_a$ (inverse depth)', ha='center', fontsize=11)
    
    ax.text(5, 3.8, 'Information per aperture:', ha='center', fontsize=11)
    ax.text(5, 3.1, r'$I_a = \ln(1/s_a) = \ln(n_a)$', ha='center', fontsize=11)
    
    ax.text(5, 2.2, r'$S = k_B \sum_a I_a$', ha='center', fontsize=12)
    
    # Final result
    result_box = FancyBboxPatch((0.8, 0.5), 8.4, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='mistyrose', edgecolor='black', linewidth=2)
    ax.add_patch(result_box)
    ax.text(5, 1.1, r'$S_{part} = k_B \sum_a \ln(1/s_a)$', 
            ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Add equivalence statement at bottom
    fig.text(0.5, -0.02, 
             r'$\mathbf{EQUIVALENCE:}\ S_{cat} = S_{osc} = S_{part}$ when $n = (A/A_0)^2 = 1/s$',
             ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return fig

def save_data():
    """Save derivation data"""
    data = {
        'description': 'Three equivalent entropy derivations',
        'categorical': {
            'formula': 'S = k_B M ln n',
            'M': 'number of categories',
            'n': 'states per category',
            'total_configs': 'W = n^M'
        },
        'oscillatory': {
            'formula': 'S = k_B sum ln(A_i/A_0)^2',
            'A_i': 'amplitude of mode i',
            'A_0': 'reference (ground state) amplitude',
            'phase_space': 'Gamma = pi m omega A^2'
        },
        'partition': {
            'formula': 'S = k_B sum ln(1/s_a)',
            's_a': 'selectivity of aperture a',
            'relation': 's = 1/n (inverse of categorical depth)'
        },
        'equivalence': 'All three give identical results when n = (A/A_0)^2 = 1/s'
    }
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'figures', 'fig_entropy_derivations.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {data_path}")

if __name__ == '__main__':
    fig = create_figure()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, '..', 'figures', 'fig_entropy_derivations.png')
    fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    
    save_data()
    plt.show()

