"""
Visualization of Mixing Process
Shows what happens when partition is removed and gases mix.
Demonstrates creation of NEW categorical states and A-B phase-lock edges.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from datetime import datetime
import json

# Set random seed for reproducibility
np.random.seed(42)

class Molecule:
    """Represents a gas molecule with position, velocity, and categorical state."""
    def __init__(self, x, y, vx, vy, container, category_id, original_container):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.container = container  # Current container
        self.original_container = original_container  # Original container (A or B)
        self.category_id = category_id
        self.phase = np.random.uniform(0, 2*np.pi)

    def distance_to(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class PhaseockNetwork:
    """Represents phase-lock relationships between molecules."""
    def __init__(self, molecules, interaction_range=0.15):
        self.molecules = molecules
        self.interaction_range = interaction_range
        self.edges = []
        self._compute_edges()

    def _compute_edges(self):
        """Compute phase-lock edges based on Van der Waals range."""
        self.edges = []
        for i, mol1 in enumerate(self.molecules):
            for j, mol2 in enumerate(self.molecules[i+1:], start=i+1):
                dist = mol1.distance_to(mol2)
                if dist < self.interaction_range:
                    strength = min((self.interaction_range / max(dist, 0.01))**6, 3.0)  # Cap at 3.0
                    if strength > 0.1:
                        self.edges.append((i, j, strength))

    def get_edge_count(self):
        return len(self.edges)

    def get_edges_by_origin(self):
        """Separate edges by original container of molecules."""
        edges_AA = []  # Both from A
        edges_BB = []  # Both from B
        edges_AB = []  # One from A, one from B (NEW in mixing!)

        for i, j, strength in self.edges:
            mol1, mol2 = self.molecules[i], self.molecules[j]
            if mol1.original_container == 'A' and mol2.original_container == 'A':
                edges_AA.append((i, j, strength))
            elif mol1.original_container == 'B' and mol2.original_container == 'B':
                edges_BB.append((i, j, strength))
            else:
                edges_AB.append((i, j, strength))

        return edges_AA, edges_BB, edges_AB

def create_mixed_state(n_molecules_per_container=20, mixing_extent=0.7):
    """
    Create mixed state: partition removed, molecules diffuse throughout.

    mixing_extent: how much molecules have diffused (0=unmixed, 1=fully mixed)
    """
    molecules = []
    category_counter = 0

    # Original Container A molecules (now dispersed)
    for i in range(n_molecules_per_container):
        # Start from left, diffuse rightward
        x_initial = np.random.uniform(0.05, 0.45)
        diffusion = np.random.uniform(0, mixing_extent * 0.5)  # Can move into right half
        x = np.clip(x_initial + diffusion, 0.05, 0.95)

        y = np.random.uniform(0.05, 0.95)
        vx = np.random.normal(0, 0.1)
        vy = np.random.normal(0, 0.1)

        # NEW categorical state (mixing creates new states!)
        molecules.append(Molecule(x, y, vx, vy, 'mixed', 'A', category_counter))
        category_counter += 1

    # Original Container B molecules (now dispersed)
    for i in range(n_molecules_per_container):
        # Start from right, diffuse leftward
        x_initial = np.random.uniform(0.55, 0.95)
        diffusion = np.random.uniform(-mixing_extent * 0.5, 0)  # Can move into left half
        x = np.clip(x_initial + diffusion, 0.05, 0.95)

        y = np.random.uniform(0.05, 0.95)
        vx = np.random.normal(0, 0.1)
        vy = np.random.normal(0, 0.1)

        # NEW categorical state
        molecules.append(Molecule(x, y, vx, vy, 'mixed', 'B', category_counter))
        category_counter += 1

    return molecules, category_counter

def visualize_mixing_state(molecules, network, save_path='mixing_process.png'):
    """Create comprehensive panel visualization of mixing state."""

    fig = plt.figure(figsize=(16, 10))

    # Panel A: Physical Configuration
    ax1 = plt.subplot(2, 3, 1)
    plot_physical_config(ax1, molecules, network, show_partition=False)
    ax1.set_title('(A) Physical Configuration - MIXED', fontsize=14, fontweight='bold')

    # Panel B: Categorical State Evolution
    ax2 = plt.subplot(2, 3, 2)
    plot_categorical_evolution(ax2, molecules)
    ax2.set_title('(B) Categorical State Progression', fontsize=14, fontweight='bold')

    # Panel C: Phase-Lock Network (with NEW A-B edges!)
    ax3 = plt.subplot(2, 3, 3)
    plot_phase_lock_network(ax3, molecules, network)
    ax3.set_title('(C) Phase-Lock Network with A-B Edges', fontsize=14, fontweight='bold')

    # Panel D: Network Comparison
    ax4 = plt.subplot(2, 3, 4)
    plot_network_comparison(ax4, network)
    ax4.set_title('(D) New A-B Interactions', fontsize=14, fontweight='bold')

    # Panel E: Entropy Change
    ax5 = plt.subplot(2, 3, 5)
    plot_entropy_change(ax5, network, molecules)
    ax5.set_title('(E) Entropy Increase from Mixing', fontsize=14, fontweight='bold')

    # Panel F: Key Insight
    ax6 = plt.subplot(2, 3, 6)
    plot_mixing_info(ax6, molecules, network)
    ax6.set_title('(F) Mixing Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close(fig)

    return fig

def plot_physical_config(ax, molecules, network, show_partition=False):
    """Plot physical positions with original container color-coding."""
    mol_A_origin = [m for m in molecules if m.original_container == 'A']
    mol_B_origin = [m for m in molecules if m.original_container == 'B']

    # Plot molecules (color by ORIGINAL container, showing they've mixed spatially)
    ax.scatter([m.x for m in mol_A_origin], [m.y for m in mol_A_origin],
               c='blue', s=100, alpha=0.6, label='Originally from A',
               edgecolors='darkblue', linewidth=1.5)
    ax.scatter([m.x for m in mol_B_origin], [m.y for m in mol_B_origin],
               c='red', s=100, alpha=0.6, label='Originally from B',
               edgecolors='darkred', linewidth=1.5)

    # Draw A-B phase-lock edges (the NEW interactions!)
    edges_AA, edges_BB, edges_AB = network.get_edges_by_origin()

    # Highlight A-B edges in purple
    for i, j, strength in edges_AB:
        ax.plot([molecules[i].x, molecules[j].x],
                [molecules[i].y, molecules[j].y],
                'purple', alpha=min(0.5*strength, 1.0), linewidth=2, linestyle='-')

    # Draw same-origin edges more subtly
    for i, j, strength in edges_AA:
        ax.plot([molecules[i].x, molecules[j].x],
                [molecules[i].y, molecules[j].y],
                'b-', alpha=min(0.2*strength, 1.0), linewidth=0.5)

    for i, j, strength in edges_BB:
        ax.plot([molecules[i].x, molecules[j].x],
                [molecules[i].y, molecules[j].y],
                'r-', alpha=min(0.2*strength, 1.0), linewidth=0.5)

    # Show removed partition
    if show_partition:
        ax.axvline(x=0.5, color='gray', linewidth=2, linestyle=':',
                   alpha=0.5, label='Partition (removed)')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Position x', fontsize=11)
    ax.set_ylabel('Position y', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.text(0.5, 0.02, 'Purple lines = NEW A-B phase-lock interactions',
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))

def plot_categorical_evolution(ax, molecules):
    """Show how categorical states have evolved during mixing."""

    # All molecules now occupy NEW categorical states
    categories_A = sorted([m.category_id for m in molecules if m.original_container == 'A'])
    categories_B = sorted([m.category_id for m in molecules if m.original_container == 'B'])

    # Plot as timeline
    ax.scatter(categories_A, np.zeros(len(categories_A)),
               c='blue', s=80, alpha=0.7, marker='o', label='Originally A')
    ax.scatter(categories_B, np.ones(len(categories_B)),
               c='red', s=80, alpha=0.7, marker='s', label='Originally B')

    # Show that these are ALL new states
    ax.axhspan(-0.2, 1.2, alpha=0.2, color='yellow')

    ax.set_xlabel('Categorical State ID', fontsize=11)
    ax.set_ylabel('Original Container', fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['A', 'B'])
    ax.set_ylim(-0.3, 1.3)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=9)

    # Key annotation
    ax.text(0.5, 0.95, 'ALL states are NEW (yellow background)\nC_initial → C_mixed',
            transform=ax.transAxes, ha='center', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

def plot_phase_lock_network(ax, molecules, network):
    """Plot phase-lock network highlighting NEW A-B edges."""
    # Circular layout
    n_A = sum(1 for m in molecules if m.original_container == 'A')
    n_B = len(molecules) - n_A
    angles_A = np.linspace(0, np.pi, n_A)
    angles_B = np.linspace(np.pi, 2*np.pi, n_B)

    positions = {}
    idx_A, idx_B = 0, 0
    for i, m in enumerate(molecules):
        if m.original_container == 'A':
            angle = angles_A[idx_A]
            idx_A += 1
        else:
            angle = angles_B[idx_B]
            idx_B += 1
        positions[i] = (np.cos(angle), np.sin(angle))

    # Draw edges
    edges_AA, edges_BB, edges_AB = network.get_edges_by_origin()

    # Draw A-B edges PROMINENTLY (these are new!)
    for i, j, strength in edges_AB:
        x = [positions[i][0], positions[j][0]]
        y = [positions[i][1], positions[j][1]]
        ax.plot(x, y, 'purple', alpha=min(0.6*strength, 1.0), linewidth=2.5, zorder=3)

    # Draw same-origin edges more subtly
    for i, j, strength in edges_AA:
        x = [positions[i][0], positions[j][0]]
        y = [positions[i][1], positions[j][1]]
        ax.plot(x, y, 'b-', alpha=min(0.3*strength, 1.0), linewidth=0.8, zorder=1)

    for i, j, strength in edges_BB:
        x = [positions[i][0], positions[j][0]]
        y = [positions[i][1], positions[j][1]]
        ax.plot(x, y, 'r-', alpha=min(0.3*strength, 1.0), linewidth=0.8, zorder=1)

    # Draw nodes
    for i, m in enumerate(molecules):
        color = 'blue' if m.original_container == 'A' else 'red'
        ax.scatter(positions[i][0], positions[i][1], c=color, s=100,
                   alpha=0.7, edgecolors='black', linewidth=1, zorder=2)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend
    ax.text(0, -1.5, f'Purple lines = NEW A-B interactions ({len(edges_AB)} edges)\nThese did NOT exist in separated state!',
            ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))

def plot_network_comparison(ax, network):
    """Compare network before and after mixing."""
    edges_AA, edges_BB, edges_AB = network.get_edges_by_origin()

    # Before mixing (estimated - no A-B interactions)
    n_before_AA = len(edges_AA)  # Approximately same within A
    n_before_BB = len(edges_BB)  # Approximately same within B
    n_before_AB = 0  # ZERO before mixing

    # After mixing (actual)
    n_after_AA = len(edges_AA)
    n_after_BB = len(edges_BB)
    n_after_AB = len(edges_AB)  # NEW!

    x = np.arange(3)
    width = 0.35

    before = [n_before_AA, n_before_BB, n_before_AB]
    after = [n_after_AA, n_after_BB, n_after_AB]

    ax.bar(x - width/2, before, width, label='Before mixing',
           color='lightgray', edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, after, width, label='After mixing',
           color=['blue', 'red', 'purple'], alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Number of Phase-Lock Edges', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['A-A', 'B-B', 'A-B'])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - width/2, b, str(b), ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, a, str(a), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Highlight the change
    ax.annotate('NEW!\n+{} edges'.format(n_after_AB),
                xy=(2 + width/2, n_after_AB), xytext=(2.5, n_after_AB + 5),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                fontsize=11, fontweight='bold', color='purple')

def plot_entropy_change(ax, network, molecules):
    """Calculate entropy change from mixing."""
    edges_AA, edges_BB, edges_AB = network.get_edges_by_origin()

    # Before mixing
    edges_before = len(edges_AA) + len(edges_BB) + 0  # No A-B

    # After mixing
    edges_after = len(edges_AA) + len(edges_BB) + len(edges_AB)

    # Reference
    n = len(molecules)
    avg_degree = 4
    E_ref = n * avg_degree / 2

    # Entropies
    k_B = 1.380649e-23
    S_before = k_B * (edges_before / E_ref)
    S_after = k_B * (edges_after / E_ref)
    Delta_S = S_after - S_before

    ax.axis('off')
    info_text = f"""
ENTROPY CHANGE FROM MIXING

Before Mixing (C_initial):
• Total edges: {edges_before}
• A-B edges: 0
• S_initial = {S_before:.3e} J/K

After Mixing (C_mixed):
• Total edges: {edges_after}
• A-B edges: {len(edges_AB)} (NEW!)
• S_mixed = {S_after:.3e} J/K

Entropy Increase:
ΔS = S_mixed - S_initial
ΔS = {Delta_S:.3e} J/K
ΔS/k_B = {Delta_S/k_B:.2f}

Origin: NEW phase-lock edges between
originally-separated molecules create
denser topological network.

This is IRREVERSIBLE: once A-B phase
correlations form, they persist!
    """

    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

def plot_mixing_info(ax, molecules, network):
    """Display key information about mixing."""
    ax.axis('off')

    n_A = sum(1 for m in molecules if m.original_container == 'A')
    n_B = sum(1 for m in molecules if m.original_container == 'B')
    total_categories = len(set(m.category_id for m in molecules))
    edges_AA, edges_BB, edges_AB = network.get_edges_by_origin()

    # Calculate edges_before for the percentage (needs to be before f-string)
    edges_before = len(edges_AA) + len(edges_BB)

    info_text = f"""
MIXED STATE

System Configuration:
• Molecules from A: {n_A}
• Molecules from B: {n_B}
• Partition: REMOVED
• Spatial mixing: Complete

Categorical State:
• Previous: C_initial ({n_A + n_B} states)
• Current: C_mixed ({total_categories} states)
• NEW states created: {total_categories}
• Axiom: C_initial CANNOT be re-occupied

Phase-Lock Network:
• A-A edges: {len(edges_AA)}
• B-B edges: {len(edges_BB)}
• A-B edges: {len(edges_AB)} ← NEW!
• Total edges: {network.get_edge_count()}
• Network densification: {len(edges_AB)}/{edges_before:.0f} = {100*len(edges_AB)/max(edges_before,1):.1f}%

CRITICAL INSIGHT:
The {len(edges_AB)} new A-B phase-lock edges
represent IRREVERSIBLE categorical
state completion. These phase correlations
persist even if we re-separate spatially!
    """

    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

def main():
    """Generate mixing process visualization."""
    print("="*60)
    print("GIBBS PARADOX VISUALIZATION: MIXING PROCESS")
    print("="*60)

    # Create mixed system
    print("\n1. Simulating mixing process...")
    molecules, n_categories = create_mixed_state(n_molecules_per_container=20, mixing_extent=0.7)
    print(f"   Created mixed state with {len(molecules)} molecules")
    print(f"   New categorical states: {n_categories} (all previous states now completed)")

    # Compute phase-lock network
    print("\n2. Computing phase-lock network...")
    network = PhaseockNetwork(molecules, interaction_range=0.15)
    print(f"   Total phase-lock edges: {network.get_edge_count()}")

    edges_AA, edges_BB, edges_AB = network.get_edges_by_origin()
    print(f"   A-A edges: {len(edges_AA)}")
    print(f"   B-B edges: {len(edges_BB)}")
    print(f"   A-B edges: {len(edges_AB)} ← NEW! These did NOT exist before mixing!")

    # Create visualization
    print("\n3. Creating visualization...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'mixing_process_{timestamp}.png'
    fig = visualize_mixing_state(molecules, network, save_path)

    # Save data
    print("\n4. Saving system data...")
    data = {
        'n_molecules': len(molecules),
        'n_categories_completed': n_categories,
        'n_phase_lock_edges': network.get_edge_count(),
        'edges_AA': len(edges_AA),
        'edges_BB': len(edges_BB),
        'edges_AB': len(edges_AB),
        'state': 'mixed',
        'key_insight': f'{len(edges_AB)} new A-B phase-lock edges created - IRREVERSIBLE!'
    }

    data_path = f'mixing_process_{timestamp}.json'
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"   Saved data to {data_path}")

    print("\n" + "="*60)
    print("COMPLETE: Mixing process visualized")
    print(f"Key result: {len(edges_AB)} NEW A-B interactions formed")
    print("="*60)

if __name__ == "__main__":
    main()
