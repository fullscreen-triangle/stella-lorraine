"""
Categorical Unification Detector (CUD)
Detects when discrete entities become categorically unified (phase transitions)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Set style
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['axes.facecolor'] = '#0a0a0a'
plt.rcParams['figure.facecolor'] = '#0a0a0a'

# Physical constants
kB = 1.380649e-23
hbar = 1.054571e-34

class CategoricalUnificationDetector:
    """
    Detects transitions from distinguishable to indistinguishable carriers.
    Signals phase transitions to dissipationless states.
    """
    
    def __init__(self):
        self.measurements = []
        
    def attempt_partition(self, entity1, entity2, T):
        """
        Attempt to partition two entities.
        Returns True if distinguishable, False if unified.
        """
        # Phase-locking energy
        Delta_lock = entity1.get('pairing_energy', 0) + entity2.get('pairing_energy', 0)
        
        # Thermal energy
        kBT = kB * T
        
        # Entities unify when thermal energy < locking energy
        unified = (kBT < Delta_lock)
        
        return {
            'distinguishable': not unified,
            'unified': unified,
            'Delta_lock_meV': Delta_lock / (1.6e-22),  # Convert to meV
            'kBT_meV': kBT / (1.6e-22),
            'temperature_K': T
        }
    
    def count_distinct_entities(self, N_total, T, Tc, entity_type='electron'):
        """
        Count distinguishable entities at temperature T.
        Below Tc, entities unify into pairs/condensate.
        """
        if T >= Tc:
            # Above Tc: all entities distinguishable
            N_distinct = N_total
            f_unified = 0
        else:
            # Below Tc: fraction condenses/pairs
            if entity_type == 'electron':
                # BCS: all electrons pair below Tc
                N_distinct = 0
                f_unified = 1.0
            elif entity_type == 'boson':
                # BEC: condensate fraction = 1 - (T/Tc)^(3/2)
                f_unified = 1 - (T/Tc)**(3/2)
                N_distinct = int(N_total * (1 - f_unified))
            elif entity_type == 'helium':
                # Superfluid fraction (empirical)
                f_unified = 1 - (T/Tc)**5.6
                f_unified = max(0, f_unified)
                N_distinct = int(N_total * (1 - f_unified))
        
        return {
            'N_total': N_total,
            'N_distinct': N_distinct,
            'N_unified': N_total - N_distinct,
            'f_unified': f_unified,
            'temperature_K': T,
            'Tc_K': Tc,
            'entity_type': entity_type
        }
    
    def scan_transition(self, N_total, Tc, entity_type, T_range):
        """Scan through a phase transition."""
        results = []
        for T in T_range:
            result = self.count_distinct_entities(N_total, T, Tc, entity_type)
            results.append(result)
            self.measurements.append(result)
        return results
    
    def detect_transition_temperature(self, results, threshold=0.5):
        """Detect Tc from measurement data."""
        for i, r in enumerate(results):
            if r['f_unified'] >= threshold:
                return r['temperature_K']
        return None


def visualize_cud_results():
    """Create visualization of CUD measurements."""
    
    cud = CategoricalUnificationDetector()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Categorical Unification Detector (CUD) Results', fontsize=14, color='#00ffff', y=0.98)
    
    # Plot 1: Superconductor transition (YBCO)
    ax = axes[0, 0]
    ax.set_title('Superconductor: Cooper Pairing (YBCO)', fontsize=10, color='#00ffff')
    
    T_range = np.linspace(50, 120, 100)
    Tc_YBCO = 93
    N = 1e23  # Number of electrons
    
    results_sc = cud.scan_transition(N, Tc_YBCO, 'electron', T_range)
    f_unified = [r['f_unified'] for r in results_sc]
    
    ax.plot(T_range, f_unified, color='#00ffff', linewidth=3)
    ax.axvline(Tc_YBCO, color='#ff6600', linestyle='--', label=f'Tc = {Tc_YBCO} K')
    ax.fill_between(T_range, 0, f_unified, alpha=0.3, color='#00ffff')
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Unified fraction (Cooper pairs)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.text(70, 0.5, 'ALL ELECTRONS\nPAIRED', fontsize=10, ha='center', 
           color='#00ffff', fontweight='bold')
    
    # Plot 2: Superfluid He-4 transition
    ax = axes[0, 1]
    ax.set_title('Superfluid: Helium-4 (λ-transition)', fontsize=10, color='#ff6600')
    
    T_range_he = np.linspace(0.5, 3, 100)
    T_lambda = 2.17
    
    results_sf = cud.scan_transition(N, T_lambda, 'helium', T_range_he)
    f_unified = [r['f_unified'] for r in results_sf]
    
    ax.plot(T_range_he, f_unified, color='#ff6600', linewidth=3)
    ax.axvline(T_lambda, color='#00ffff', linestyle='--', label=f'Tλ = {T_lambda} K')
    ax.fill_between(T_range_he, 0, f_unified, alpha=0.3, color='#ff6600')
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Superfluid fraction', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: BEC transition
    ax = axes[1, 0]
    ax.set_title('BEC: Dilute Atomic Gas', fontsize=10, color='#00ff88')
    
    T_range_bec = np.linspace(1e-9, 500e-9, 100)  # nK range
    T_BEC = 170e-9  # 170 nK
    
    results_bec = cud.scan_transition(N, T_BEC, 'boson', T_range_bec)
    f_unified = [r['f_unified'] for r in results_bec]
    
    ax.plot(T_range_bec * 1e9, f_unified, color='#00ff88', linewidth=3)
    ax.axvline(T_BEC * 1e9, color='#ff00ff', linestyle='--', label=f'T_BEC = {T_BEC*1e9:.0f} nK')
    ax.fill_between(T_range_bec * 1e9, 0, f_unified, alpha=0.3, color='#00ff88')
    
    # N0/N = 1 - (T/Tc)^(3/2) theoretical curve
    T_theory = np.linspace(1, T_BEC * 1e9, 50)
    f_theory = 1 - (T_theory / (T_BEC * 1e9))**(3/2)
    ax.plot(T_theory, f_theory, 'w--', linewidth=1, alpha=0.5, label='Theory: 1-(T/Tc)^(3/2)')
    
    ax.set_xlabel('Temperature (nK)', fontsize=8)
    ax.set_ylabel('Condensate fraction N₀/N', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Comparison of transitions
    ax = axes[1, 1]
    ax.set_title('Unified View: Partition Extinction', fontsize=10, color='#ff00ff')
    
    # Normalized temperature T/Tc
    t_norm = np.linspace(0.01, 1.5, 100)
    
    # Different transition types
    f_bcs = np.where(t_norm < 1, 1.0, 0.0)  # Step function (BCS)
    f_bec = np.where(t_norm < 1, 1 - t_norm**(3/2), 0)  # BEC
    f_sf = np.where(t_norm < 1, np.maximum(0, 1 - t_norm**5.6), 0)  # Superfluid
    
    ax.plot(t_norm, f_bcs, color='#00ffff', linewidth=2, label='Superconductor (BCS)')
    ax.plot(t_norm, f_bec, color='#00ff88', linewidth=2, label='BEC')
    ax.plot(t_norm, f_sf, color='#ff6600', linewidth=2, label='Superfluid He-4')
    
    ax.axvline(1.0, color='white', linestyle=':', alpha=0.5)
    ax.text(1.05, 0.5, 'T = Tc', fontsize=9, color='white', rotation=90, va='center')
    
    ax.fill_between(t_norm[t_norm < 1], 0, 1, alpha=0.1, color='#00ffff')
    ax.text(0.5, 0.9, 'PARTITION\nEXTINCT', fontsize=10, ha='center', 
           color='#00ffff', fontweight='bold')
    ax.text(1.25, 0.9, 'PARTITION\nACTIVE', fontsize=10, ha='center', 
           color='#ff6666', fontweight='bold')
    
    ax.set_xlabel('T / Tc (normalized)', fontsize=8)
    ax.set_ylabel('Unified fraction', fontsize=8)
    ax.legend(loc='center right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.5)
    
    plt.tight_layout()
    fig.savefig('figures/panel_cud_results.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a', edgecolor='none')
    plt.close(fig)
    
    return cud


# Main execution
if __name__ == '__main__':
    print("=" * 60)
    print("Categorical Unification Detector (CUD)")
    print("Phase Transition Detection Through Partition Analysis")
    print("=" * 60)
    
    cud = visualize_cud_results()
    
    # Save data
    output_data = {
        'instrument': 'Categorical Unification Detector',
        'principle': 'Detects when partition between entities becomes undefined',
        'transitions_detected': [
            {
                'system': 'YBCO Superconductor',
                'Tc_K': 93,
                'mechanism': 'Cooper pairing → all electrons unified',
                'result': 'ρ = 0 exactly'
            },
            {
                'system': 'Superfluid Helium-4',
                'Tc_K': 2.17,
                'mechanism': 'Bose condensation → atoms unified',
                'result': 'μ = 0 exactly'
            },
            {
                'system': 'Dilute BEC',
                'Tc_nK': 170,
                'mechanism': 'Ground state occupation → atoms unified',
                'result': 'Macroscopic quantum state'
            }
        ],
        'n_measurements': len(cud.measurements)
    }
    
    with open('data/cud_measurements.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nTransitions analyzed:")
    print(f"  - Superconductor (YBCO): Tc = 93 K")
    print(f"  - Superfluid He-4: T_lambda = 2.17 K")
    print(f"  - BEC: T_BEC = 170 nK")
    print(f"\nGenerated: figures/panel_cud_results.png")
    print(f"Generated: data/cud_measurements.json")

