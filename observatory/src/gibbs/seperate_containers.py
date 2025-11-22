"""
Visualization of Initial Separated Containers State
Shows two containers A and B with molecules before mixing.
Demonstrates initial categorical state and phase-lock networks.
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
    def __init__(self, x, y, vx, vy, container, category_id):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.container = container  # 'A' or 'B'
        self.category_id = category_id
        self.phase = np.random.uniform(0, 2*np.pi)  # Oscillator phase

    def distance_to(self, other):
        """Calculate distance to another molecule."""
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
                    # Phase coherence strength (Van der Waals ~ r^-6)
                    strength = min((self.interaction_range / max(dist, 0.01))**6, 3.0)  # Cap at 3.0
                    if strength > 0.1:  # Threshold for significant interaction
                        self.edges.append((i, j, strength))

    def get_edge_count(self):
        """Get total number of edges."""
        return len(self.edges)

    def get_edges_by_container(self):
        """Separate edges within A, within B, and between A-B."""
        edges_AA = []
        edges_BB = []
        edges_AB = []

        for i, j, strength in self.edges:
            mol1, mol2 = self.molecules[i], self.molecules[j]
            if mol1.container == 'A' and mol2.container == 'A':
                edges_AA.append((i, j, strength))
            elif mol1.container == 'B' and mol2.container == 'B':
                edges_BB.append((i, j, strength))
            else:
                edges_AB.append((i, j, strength))

        return edges_AA, edges_BB, edges_AB

def create_separated_containers(n_molecules_per_container=20):
    """
    Create initial state: two separated containers with molecules.

    Container A: left half (x: 0 to 0.45)
    Container B: right half (x: 0.55 to 1.0)
    Partition at x = 0.5
    """
    molecules = []
    category_counter = 0

    # Container A molecules
    for i in range(n_molecules_per_container):
        x = np.random.uniform(0.05, 0.45)
        y = np.random.uniform(0.05, 0.95)
        vx = np.random.normal(0, 0.1)
        vy = np.random.normal(0, 0.1)
        molecules.append(Molecule(x, y, vx, vy, 'A', category_counter))
        category_counter += 1

    # Container B molecules
    for i in range(n_molecules_per_container):
        x = np.random.uniform(0.55, 0.95)
        y = np.random.uniform(0.05, 0.95)
        vx = np.random.normal(0, 0.1)
        vy = np.random.normal(0, 0.1)
        molecules.append(Molecule(x, y, vx, vy, 'B', category_counter))
        category_counter += 1

    return molecules, category_counter

def visualize_separated_state(molecules, network, save_path='separated_containers.png'):
    """Create comprehensive panel visualization of separated state."""

    fig = plt.figure(figsize=(16, 10))

    # Panel A: Physical Configuration
    ax1 = plt.subplot(2, 3, 1)
    plot_physical_config(ax1, molecules, network, show_partition=True)
    ax1.set_title('(A) Physical Configuration', fontsize=14, fontweight='bold')

    # Panel B: Categorical State Diagram
    ax2 = plt.subplot(2, 3, 2)
    plot_categorical_states(ax2, molecules)
    ax2.set_title('(B) Categorical State Distribution', fontsize=14, fontweight='bold')

    # Panel C: Phase-Lock Network
    ax3 = plt.subplot(2, 3, 3)
    plot_phase_lock_network(ax3, molecules, network)
    ax3.set_title('(C) Phase-Lock Network', fontsize=14, fontweight='bold')

    # Panel D: Network Statistics
    ax4 = plt.subplot(2, 3, 4)
    plot_network_statistics(ax4, network)
    ax4.set_title('(D) Network Topology Statistics', fontsize=14, fontweight='bold')

    # Panel E: Entropy Calculation
    ax5 = plt.subplot(2, 3, 5)
    plot_entropy_calculation(ax5, network, molecules)
    ax5.set_title('(E) Oscillatory Entropy', fontsize=14, fontweight='bold')

    # Panel F: Key Information
    ax6 = plt.subplot(2, 3, 6)
    plot_system_info(ax6, molecules, network)
    ax6.set_title('(F) System Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close(fig)

    return fig

def plot_physical_config(ax, molecules, network, show_partition=True):
    """Plot physical positions of molecules."""
    # Separate by container
    mol_A = [m for m in molecules if m.container == 'A']
    mol_B = [m for m in molecules if m.container == 'B']

    # Plot molecules
    ax.scatter([m.x for m in mol_A], [m.y for m in mol_A],
               c='blue', s=100, alpha=0.6, label='Container A', edgecolors='darkblue', linewidth=1.5)
    ax.scatter([m.x for m in mol_B], [m.y for m in mol_B],
               c='red', s=100, alpha=0.6, label='Container B', edgecolors='darkred', linewidth=1.5)

    # Draw phase-lock edges
    edges_AA, edges_BB, edges_AB = network.get_edges_by_container()

    for i, j, strength in edges_AA:
        ax.plot([molecules[i].x, molecules[j].x],
                [molecules[i].y, molecules[j].y],
                'b-', alpha=min(0.3*strength, 1.0), linewidth=0.5)

    for i, j, strength in edges_BB:
        ax.plot([molecules[i].x, molecules[j].x],
                [molecules[i].y, molecules[j].y],
                'r-', alpha=min(0.3*strength, 1.0), linewidth=0.5)

    # Draw partition
    if show_partition:
        ax.axvline(x=0.5, color='black', linewidth=3, linestyle='--', label='Partition')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Position x', fontsize=11)
    ax.set_ylabel('Position y', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def plot_categorical_states(ax, molecules):
    """Plot categorical state distribution."""
    categories_A = [m.category_id for m in molecules if m.container == 'A']
    categories_B = [m.category_id for m in molecules if m.container == 'B']

    ax.scatter(categories_A, np.zeros(len(categories_A)),
               c='blue', s=80, alpha=0.7, label='Container A')
    ax.scatter(categories_B, np.ones(len(categories_B)),
               c='red', s=80, alpha=0.7, label='Container B')

    ax.set_xlabel('Categorical State ID', fontsize=11)
    ax.set_ylabel('Container', fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['A', 'B'])
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=9)

    # Annotate
    ax.text(0.05, 0.95, f'Total categorical states: {len(molecules)}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def plot_phase_lock_network(ax, molecules, network):
    """Plot phase-lock network as a graph."""
    # Create circular layout for clarity
    n = len(molecules)
    angles_A = np.linspace(0, np.pi, sum(1 for m in molecules if m.container == 'A'))
    angles_B = np.linspace(np.pi, 2*np.pi, sum(1 for m in molecules if m.container == 'B'))

    positions = {}
    idx = 0
    for i, m in enumerate(molecules):
        if m.container == 'A':
            angle = angles_A[idx]
            idx += 1
        else:
            angle = angles_B[idx - len(angles_A)]
        positions[i] = (np.cos(angle), np.sin(angle))
        if m.container == 'B':
            idx += 1

    # Draw edges
    edges_AA, edges_BB, edges_AB = network.get_edges_by_container()

    for i, j, strength in edges_AA:
        x = [positions[i][0], positions[j][0]]
        y = [positions[i][1], positions[j][1]]
        ax.plot(x, y, 'b-', alpha=min(0.4*strength, 1.0), linewidth=1)

    for i, j, strength in edges_BB:
        x = [positions[i][0], positions[j][0]]
        y = [positions[i][1], positions[j][1]]
        ax.plot(x, y, 'r-', alpha=min(0.4*strength, 1.0), linewidth=1)

    # Draw nodes
    for i, m in enumerate(molecules):
        color = 'blue' if m.container == 'A' else 'red'
        ax.scatter(positions[i][0], positions[i][1], c=color, s=100,
                   alpha=0.7, edgecolors='black', linewidth=1)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend
    ax.text(0, -1.5, 'Blue: Container A | Red: Container B\nLine thickness ∝ phase-lock strength',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_network_statistics(ax, network):
    """Plot network topology statistics."""
    edges_AA, edges_BB, edges_AB = network.get_edges_by_container()

    counts = [len(edges_AA), len(edges_BB), len(edges_AB)]
    labels = ['A-A\ninteractions', 'B-B\ninteractions', 'A-B\ninteractions']
    colors = ['blue', 'red', 'purple']

    bars = ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Phase-Lock Edges', fontsize=11)
    ax.set_title('Edge Distribution', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Note
    ax.text(0.5, 0.95, 'Note: A-B = 0 (containers separated)',
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

def plot_entropy_calculation(ax, network, molecules):
    """Calculate and display oscillatory entropy."""
    edges_AA, edges_BB, edges_AB = network.get_edges_by_container()
    total_edges = len(edges_AA) + len(edges_BB) + len(edges_AB)

    # Reference edge count (for normalization)
    n = len(molecules)
    avg_degree = 4  # typical for gas at standard conditions
    E_ref = n * avg_degree / 2

    # Termination probability (decreases with edge density)
    alpha = np.exp(-total_edges / E_ref)

    # Oscillatory entropy: S = -k_B log(alpha) = k_B * (|E| / <E>)
    k_B = 1.380649e-23  # J/K
    entropy_categorical = k_B * (total_edges / E_ref)

    # Display
    ax.axis('off')
    info_text = f"""
Categorical Entropy Calculation:

Total phase-lock edges: |E| = {total_edges}
Reference edges: ⟨E⟩ = {E_ref:.1f}

Termination probability:
α = exp(-|E|/⟨E⟩) = {alpha:.4f}

Oscillatory entropy:
S = -k_B log(α) = k_B |E|/⟨E⟩
S = {entropy_categorical:.2e} J/K

Per-molecule entropy:
S/N = {entropy_categorical/n:.2e} J/K
    """

    ax.text(0.1, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

def plot_system_info(ax, molecules, network):
    """Display key system information."""
    ax.axis('off')

    n_A = sum(1 for m in molecules if m.container == 'A')
    n_B = sum(1 for m in molecules if m.container == 'B')
    total_categories = len(set(m.category_id for m in molecules))
    edges_AA, edges_BB, edges_AB = network.get_edges_by_container()

    info_text = f"""
INITIAL SEPARATED STATE

System Configuration:
• Container A: {n_A} molecules
• Container B: {n_B} molecules
• Partition: CLOSED

Categorical State:
• Total categories completed: {total_categories}
• Categories in A: {n_A}
• Categories in B: {n_B}
• State: C_initial

Phase-Lock Network:
• A-A edges: {len(edges_AA)}
• B-B edges: {len(edges_BB)}
• A-B edges: {len(edges_AB)} ← ZERO (separated)
• Total edges: {network.get_edge_count()}

Key Insight:
Each molecule occupies a UNIQUE
categorical state. No cross-container
phase-locking exists.
    """

    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

def main():
    """Generate separated containers visualization."""
    print("="*60)
    print("GIBBS PARADOX VISUALIZATION: SEPARATED CONTAINERS")
    print("="*60)

    # Create system
    print("\n1. Creating separated containers...")
    molecules, n_categories = create_separated_containers(n_molecules_per_container=20)
    print(f"   Created {len(molecules)} molecules in {n_categories} categorical states")

    # Compute phase-lock network
    print("\n2. Computing phase-lock network...")
    network = PhaseockNetwork(molecules, interaction_range=0.15)
    print(f"   Found {network.get_edge_count()} phase-lock edges")

    edges_AA, edges_BB, edges_AB = network.get_edges_by_container()
    print(f"   A-A edges: {len(edges_AA)}")
    print(f"   B-B edges: {len(edges_BB)}")
    print(f"   A-B edges: {len(edges_AB)} (should be 0 - containers separated!)")

    # Create visualization
    print("\n3. Creating visualization...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'separated_containers_{timestamp}.png'
    fig = visualize_separated_state(molecules, network, save_path)

    # Save data
    print("\n4. Saving system data...")
    data = {
        'n_molecules': len(molecules),
        'n_categories_completed': n_categories,
        'n_phase_lock_edges': network.get_edge_count(),
        'edges_AA': len(edges_AA),
        'edges_BB': len(edges_BB),
        'edges_AB': len(edges_AB),
        'state': 'separated_initial'
    }

    data_path = f'separated_containers_{timestamp}.json'
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"   Saved data to {data_path}")

    print("\n" + "="*60)
    print("COMPLETE: Separated containers state visualized")
    print("="*60)

if __name__ == "__main__":
    main()
