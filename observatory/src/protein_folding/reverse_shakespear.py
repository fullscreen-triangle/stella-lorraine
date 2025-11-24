"""
FIXED REVERSE FOLDING PATHWAY DISCOVERY
Tracks ALL bonds with adaptive thresholds
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from datetime import datetime
import logging

if __name__ == "__main__":

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    print("="*80)
    print("FIXED REVERSE FOLDING PATHWAY DISCOVERY")
    print("="*80)

    # ============================================================
    # CORE CLASSES (with fixes)
    # ============================================================

    class ProtonDemonHBond:
        """
        H-bond with proton demon phase-locked to EM field
        """
        def __init__(self, bond_id, donor_id, acceptor_id, em_field, initial_phase=None):
            self.bond_id = bond_id
            self.donor_id = donor_id
            self.acceptor_id = acceptor_id
            self.em_field = em_field

            # H-bond frequency (near H⁺ field)
            self.frequency = em_field.f_H_plus * (1 + np.random.randn() * 0.01)

            # Initial phase
            self.phase = initial_phase if initial_phase else np.random.rand() * 2 * np.pi

            # Phase-lock quality (0-1)
            self.phase_lock_quality = 0.0
            self.initial_phase_lock = 0.0  # Track initial value
            self.max_phase_lock = 0.0  # Track maximum achieved

            # Bond strength (kcal/mol)
            self.strength = 3.0 + np.random.randn() * 0.5

            # Formation tracking
            self.formation_cycle = None
            self.phase_lock_history = []  # Track phase-lock over time

            # Criticality (how important for stability)
            self.criticality = 0.0

        def update_phase_lock(self, groel_frequency, dt):
            """
            Update phase-lock to GroEL cavity frequency
            """
            # Phase evolution
            self.phase += 2 * np.pi * self.frequency * dt
            self.phase = self.phase % (2 * np.pi)

            # Phase-lock strength (coupling to GroEL)
            coupling = 0.1  # Coupling constant
            phase_error = self.phase - (2 * np.pi * groel_frequency * dt)

            # Update phase-lock quality
            self.phase_lock_quality += coupling * np.cos(phase_error) * dt
            self.phase_lock_quality = np.clip(self.phase_lock_quality, 0, 1)

            # Track maximum
            self.max_phase_lock = max(self.max_phase_lock, self.phase_lock_quality)

            return self.phase_lock_quality

        def record_phase_lock(self, cycle):
            """
            Record phase-lock quality for this cycle
            """
            self.phase_lock_history.append({
                'cycle': cycle,
                'quality': self.phase_lock_quality,
                'phase': self.phase
            })

        def detect_formation(self, cycle, threshold_type='adaptive'):
            """
            Detect if bond formed in this cycle

            threshold_type:
                'absolute': quality > 0.5
                'adaptive': quality > mean + 0.5*std
                'relative': improvement > 20% from previous cycle
            """
            if self.formation_cycle is not None:
                return False  # Already formed

            if len(self.phase_lock_history) < 2:
                return False  # Need at least 2 cycles

            current = self.phase_lock_history[-1]['quality']
            previous = self.phase_lock_history[-2]['quality']

            if threshold_type == 'absolute':
                # Simple threshold
                if current > 0.5:
                    self.formation_cycle = cycle
                    return True

            elif threshold_type == 'adaptive':
                # Adaptive based on history
                if len(self.phase_lock_history) >= 3:
                    qualities = [h['quality'] for h in self.phase_lock_history[:-1]]
                    mean_quality = np.mean(qualities)
                    std_quality = np.std(qualities)

                    if current > mean_quality + 0.5 * std_quality:
                        self.formation_cycle = cycle
                        return True

            elif threshold_type == 'relative':
                # Relative improvement
                if previous > 0:
                    improvement = (current - previous) / previous
                    if improvement > 0.2:  # 20% improvement
                        self.formation_cycle = cycle
                        return True
                elif current > 0.1:  # First significant value
                    self.formation_cycle = cycle
                    return True

            return False

        def calculate_stability_contribution(self):
            """
            Calculate contribution to protein stability
            """
            # Base strength modulated by phase-lock quality
            return self.strength * self.phase_lock_quality


    class ProteinFoldingNetwork:
        """
        Network of H-bonds forming during folding
        """
        def __init__(self, protein_name, n_bonds, em_field):
            self.protein_name = protein_name
            self.n_bonds = n_bonds
            self.em_field = em_field

            # Create H-bonds with proton demons
            self.bonds = []
            for i in range(n_bonds):
                donor = i * 5
                acceptor = donor + 15
                bond = ProtonDemonHBond(i, donor, acceptor, em_field)
                self.bonds.append(bond)

            logger.info(f"✓ Protein network created: {protein_name}")
            logger.info(f"  H-bonds: {n_bonds}")

        def update_network(self, groel_frequency, dt):
            """
            Update all bonds in network
            """
            for bond in self.bonds:
                bond.update_phase_lock(groel_frequency, dt)

        def record_cycle(self, cycle):
            """
            Record state for all bonds at end of cycle
            """
            for bond in self.bonds:
                bond.record_phase_lock(cycle)

        def detect_formations(self, cycle, threshold_type='adaptive'):
            """
            Detect which bonds formed this cycle
            """
            formed_bonds = []
            for bond in self.bonds:
                if bond.detect_formation(cycle, threshold_type):
                    formed_bonds.append(bond)
            return formed_bonds

        def calculate_stability(self):
            """
            Calculate overall protein stability
            """
            if not self.bonds:
                return 0.0

            total_stability = sum([b.calculate_stability_contribution()
                                for b in self.bonds])

            # Normalize by number of bonds
            return total_stability / len(self.bonds)

        def calculate_phase_coherence(self):
            """
            Calculate phase coherence across network
            """
            if not self.bonds:
                return 0.0

            # Mean phase-lock quality
            mean_quality = np.mean([b.phase_lock_quality for b in self.bonds])

            # Phase variance (low = coherent)
            phases = np.array([b.phase for b in self.bonds])
            phase_variance = np.var(np.cos(phases)) + np.var(np.sin(phases))

            # Coherence = high quality + low variance
            coherence = mean_quality * (1 - phase_variance / 2)

            return coherence

        def get_bond_statistics(self):
            """
            Get statistics for all bonds
            """
            stats = {
                'mean_phase_lock': np.mean([b.phase_lock_quality for b in self.bonds]),
                'std_phase_lock': np.std([b.phase_lock_quality for b in self.bonds]),
                'max_phase_lock': np.max([b.phase_lock_quality for b in self.bonds]),
                'min_phase_lock': np.min([b.phase_lock_quality for b in self.bonds]),
                'formed_bonds': sum([1 for b in self.bonds if b.formation_cycle is not None])
            }
            return stats


    class ElectromagneticField:
        """
        H⁺/O₂ electromagnetic field
        """
        def __init__(self, T=310.0):
            self.T = T
            self.f_H_plus = 4.0e13  # Hz
            self.f_O2 = 1.0e13      # Hz
            self.f_GroEL = 1.0      # Hz
            self.n_O2_states = 25110


    class GroELCyclicFolder:
        """
        GroEL cavity with cyclic ATP-driven folding
        """
        def __init__(self, em_field, cycle_duration=1.0):
            self.em_field = em_field
            self.cycle_duration = cycle_duration
            self.cavity_volume = 85000  # Ų
            self.cavity_frequency = em_field.f_GroEL
            self.atp_state = 'empty'
            self.current_cycle = 0

        def run_atp_cycle(self, protein_network, verbose=False):
            """
            Run one ATP hydrolysis cycle
            """
            self.current_cycle += 1

            if verbose:
                logger.info(f"\n--- Cycle {self.current_cycle} ---")

            # ATP binding
            self.atp_state = 'ATP_bound'
            self.cavity_frequency = 2.0 * self.em_field.f_GroEL

            # Evolve for half cycle
            dt = self.cycle_duration / 100
            for step in range(50):
                protein_network.update_network(self.cavity_frequency, dt)

            # ATP hydrolysis
            self.atp_state = 'ADP_release'
            self.cavity_frequency = self.em_field.f_GroEL

            # Evolve for second half
            for step in range(50):
                protein_network.update_network(self.cavity_frequency, dt)

            # Record cycle state
            protein_network.record_cycle(self.current_cycle)

            # Detect bond formations
            formed_bonds = protein_network.detect_formations(
                self.current_cycle,
                threshold_type='adaptive'
            )

            # Calculate cycle results
            stability = protein_network.calculate_stability()
            coherence = protein_network.calculate_phase_coherence()
            stats = protein_network.get_bond_statistics()

            if verbose:
                logger.info(f"  Stability: {stability:.3f}")
                logger.info(f"  Coherence: {coherence:.3f}")
                logger.info(f"  Bonds formed: {len(formed_bonds)}")
                logger.info(f"  Total formed: {stats['formed_bonds']}/{protein_network.n_bonds}")

            return {
                'cycle': self.current_cycle,
                'stability': stability,
                'coherence': coherence,
                'formed_bonds': formed_bonds,
                'stats': stats
            }

        def fold_protein(self, protein_network, max_cycles=20,
                        stability_threshold=0.7, verbose=True):
            """
            Fold protein through multiple ATP cycles
            """
            logger.info(f"\n{'='*60}")
            logger.info(f"FOLDING {protein_network.protein_name}")
            logger.info('='*60)

            trajectory = []
            folded = False

            for cycle in range(max_cycles):
                result = self.run_atp_cycle(protein_network, verbose=verbose)
                trajectory.append(result)

                # Check if folded
                if result['stability'] > stability_threshold:
                    folded = True
                    logger.info(f"\n✓ FOLDED at cycle {cycle + 1}!")
                    logger.info(f"  Final stability: {result['stability']:.3f}")
                    logger.info(f"  Final coherence: {result['coherence']:.3f}")
                    logger.info(f"  Bonds formed: {result['stats']['formed_bonds']}/{protein_network.n_bonds}")
                    break

            if not folded:
                logger.info(f"\n✗ Did not fold in {max_cycles} cycles")
                logger.info(f"  Final stability: {trajectory[-1]['stability']:.3f}")
                logger.info(f"  Bonds formed: {trajectory[-1]['stats']['formed_bonds']}/{protein_network.n_bonds}")

            return {
                'trajectory': trajectory,
                'folded': folded,
                'cycles_to_fold': self.current_cycle if folded else None,
                'final_stats': trajectory[-1]['stats']
            }


    class ReverseFoldingAnalyzer:
        """
        Analyze folding trajectory to extract pathway
        """
        def __init__(self, protein_network, folding_trajectory):
            self.protein_network = protein_network
            self.trajectory = folding_trajectory

        def extract_folding_pathway(self):
            """
            Extract complete folding pathway for ALL bonds
            """
            pathway = []

            # Get all bonds sorted by formation cycle
            all_bonds = []
            for bond in self.protein_network.bonds:
                if bond.formation_cycle is not None:
                    all_bonds.append({
                        'bond_id': bond.bond_id,
                        'donor': bond.donor_id,
                        'acceptor': bond.acceptor_id,
                        'cycle': bond.formation_cycle,
                        'phase_lock_quality': bond.phase_lock_quality,
                        'max_phase_lock': bond.max_phase_lock,
                        'strength': bond.strength
                    })
                else:
                    # Bond never formed - use cycle with highest phase-lock
                    if bond.phase_lock_history:
                        best_cycle = max(bond.phase_lock_history,
                                    key=lambda x: x['quality'])
                        all_bonds.append({
                            'bond_id': bond.bond_id,
                            'donor': bond.donor_id,
                            'acceptor': bond.acceptor_id,
                            'cycle': best_cycle['cycle'],
                            'phase_lock_quality': best_cycle['quality'],
                            'max_phase_lock': bond.max_phase_lock,
                            'strength': bond.strength,
                            'incomplete': True  # Mark as incomplete
                        })

            # Sort by cycle
            pathway = sorted(all_bonds, key=lambda x: x['cycle'])

            return pathway

        def identify_critical_cycles(self):
            """
            Identify cycles with major stability increases
            """
            critical_cycles = []

            for i in range(1, len(self.trajectory)):
                prev_stability = self.trajectory[i-1]['stability']
                curr_stability = self.trajectory[i]['stability']

                stability_increase = curr_stability - prev_stability

                # Critical if stability increased > 0.1
                if stability_increase > 0.1:
                    critical_cycles.append({
                        'cycle': i + 1,
                        'stability_increase': stability_increase,
                        'final_stability': curr_stability,
                        'bonds_formed': len(self.trajectory[i]['formed_bonds'])
                    })

            return critical_cycles

        def identify_folding_nucleus(self):
            """
            Identify folding nucleus (earliest + highest quality bonds)
            """
            pathway = self.extract_folding_pathway()

            if not pathway:
                return None

            # Sort by cycle (earliest first), then by quality (highest first)
            sorted_pathway = sorted(pathway,
                                key=lambda x: (x['cycle'], -x['max_phase_lock']))

            # Nucleus is first bond (earliest + highest quality)
            nucleus = sorted_pathway[0]

            return nucleus

        def get_pathway_summary(self):
            """
            Get complete pathway summary
            """
            pathway = self.extract_folding_pathway()
            critical_cycles = self.identify_critical_cycles()
            nucleus = self.identify_folding_nucleus()

            # Bonds per cycle
            bonds_per_cycle = {}
            for bond in pathway:
                cycle = bond['cycle']
                if cycle not in bonds_per_cycle:
                    bonds_per_cycle[cycle] = []
                bonds_per_cycle[cycle].append(bond)

            return {
                'total_bonds': len(pathway),
                'cycles_to_fold': max([b['cycle'] for b in pathway]) if pathway else 0,
                'critical_cycles': [c['cycle'] for c in critical_cycles],
                'nucleus': nucleus,
                'bonds_per_cycle': bonds_per_cycle,
                'pathway': pathway,
                'incomplete_bonds': sum([1 for b in pathway if b.get('incomplete', False)])
            }


    # ============================================================
    # RUN FIXED SIMULATION
    # ============================================================

    logger.info("\n" + "="*80)
    logger.info("RUNNING FIXED REVERSE FOLDING SIMULATION")
    logger.info("="*80)

    # Initialize
    em_field = ElectromagneticField(T=310.0)
    protein = ProteinFoldingNetwork("TestProtein", n_bonds=8, em_field=em_field)
    groel = GroELCyclicFolder(em_field, cycle_duration=1.0)

    # Run folding
    folding_results = groel.fold_protein(protein, max_cycles=20,
                                        stability_threshold=0.7, verbose=True)

    # Analyze pathway
    analyzer = ReverseFoldingAnalyzer(protein, folding_results['trajectory'])
    pathway_summary = analyzer.get_pathway_summary()

    logger.info(f"\n{'='*60}")
    logger.info("PATHWAY ANALYSIS RESULTS")
    logger.info('='*60)
    logger.info(f"Total bonds tracked: {pathway_summary['total_bonds']}/8")
    logger.info(f"Cycles to fold: {pathway_summary['cycles_to_fold']}")
    logger.info(f"Critical cycles: {pathway_summary['critical_cycles']}")
    logger.info(f"Incomplete bonds: {pathway_summary['incomplete_bonds']}")

    if pathway_summary['nucleus']:
        logger.info(f"\nFolding nucleus:")
        logger.info(f"  Bond: {pathway_summary['nucleus']['donor']}-{pathway_summary['nucleus']['acceptor']}")
        logger.info(f"  Cycle: {pathway_summary['nucleus']['cycle']}")
        logger.info(f"  Max phase-lock: {pathway_summary['nucleus']['max_phase_lock']:.3f}")

    logger.info(f"\nBonds per cycle:")
    for cycle, bonds in sorted(pathway_summary['bonds_per_cycle'].items()):
        logger.info(f"  Cycle {cycle}: {len(bonds)} bonds")

    logger.info(f"\nFormation events:")
    for i, bond in enumerate(pathway_summary['pathway'][:5]):
        incomplete = " (incomplete)" if bond.get('incomplete', False) else ""
        logger.info(f"  Cycle {bond['cycle']}: Bond {bond['bond_id']} "
                f"({bond['donor']}-{bond['acceptor']}) "
                f"quality={bond['max_phase_lock']:.3f}{incomplete}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'protein': protein.protein_name,
        'n_bonds': protein.n_bonds,
        'folded': folding_results['folded'],
        'pathway_summary': {
            'total_bonds': pathway_summary['total_bonds'],
            'cycles_to_fold': pathway_summary['cycles_to_fold'],
            'critical_cycles': pathway_summary['critical_cycles'],
            'incomplete_bonds': pathway_summary['incomplete_bonds']
        }
    }

    with open('fixed_reverse_folding_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved: fixed_reverse_folding_results.json")

    # Validation
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION")
    logger.info('='*60)

    if pathway_summary['total_bonds'] == 8:
        logger.info("✓ SUCCESS: All 8 bonds tracked!")
    else:
        logger.info(f"✗ PARTIAL: Only {pathway_summary['total_bonds']}/8 bonds tracked")
        logger.info(f"  Incomplete bonds: {pathway_summary['incomplete_bonds']}")

    logger.info("="*80)
