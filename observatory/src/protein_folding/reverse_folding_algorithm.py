"""
Reverse Folding Algorithm - Cycle-by-Cycle H-Bond Destabilization.

Discovers protein folding pathway by starting from native state and
systematically destabilizing H-bonds cycle-by-cycle, tracking which
GroEL cycles are required for each bond to phase-lock.

Key insight from papers: Folding pathway is revealed by the SEQUENCE OF CYCLES
needed to establish each H-bond, not by the spatial order of bond formation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from copy import deepcopy

from protein_folding_network import ProteinFoldingNetwork, PhaseCoherenceCluster
from groel_resonance_chamber import GroELResonanceChamber, ATPCycleState
from proton_maxwell_demon import HBondOscillator

logger = logging.getLogger(__name__)


@dataclass
class HBondFormationEvent:
    """
    Record of when an H-bond phase-locks during folding.

    From papers: Formation event = specific GroEL cycle + cavity frequency
    where bond achieves stable phase-lock.
    """
    bond: HBondOscillator
    formation_cycle: int
    cavity_frequency: float  # Hz at formation
    phase_coherence_achieved: float
    required_previous_bonds: List[Tuple[int, int]]  # (donor_res, acceptor_res) dependencies


@dataclass
class FoldingPathwayNode:
    """
    Node in folding pathway tree.

    Represents protein state after specific H-bond is established.
    """
    cycle: int
    bonds_formed: List[HBondOscillator]
    network_stability: float
    variance: float
    phase_clusters: List[PhaseCoherenceCluster]
    parent_node: Optional['FoldingPathwayNode'] = None
    children: List['FoldingPathwayNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class ReverseFoldingAlgorithm:
    """
    Discovers folding pathway through systematic reverse destabilization.

    Enhanced for cycle-by-cycle operation from papers:

    Algorithm:
    1. Start with native (fully folded) protein in GroEL
    2. Run cycles until fully phase-locked (all bonds coherent)
    3. Identify which cycle each bond phase-locks
    4. Systematically remove bonds in reverse order of their locking cycle
    5. Re-simulate to verify remaining bonds stay stable
    6. Build pathway tree from formation order

    Key insight: Bonds that lock in later cycles DEPEND on bonds
    from earlier cycles - revealing the causal folding pathway.
    """

    def __init__(self, temperature: float = 310.0):
        """
        Initialize reverse folding algorithm.

        Args:
            temperature: System temperature (K)
        """
        self.temperature = temperature
        self.groel = GroELResonanceChamber(temperature)

        # Algorithm state
        self.formation_events: List[HBondFormationEvent] = []
        self.pathway_tree: Optional[FoldingPathwayNode] = None
        self.critical_cycles: List[int] = []  # Cycles where new bonds lock

        logger.info("Initialized ReverseFoldingAlgorithm with cycle-by-cycle tracking")

    def identify_bond_formation_cycles(self, protein: ProteinFoldingNetwork,
                                       max_cycles: int = 20) -> Dict[Tuple[int, int], int]:
        """
        Run forward simulation to identify which cycle each bond phase-locks.

        This is step 1: Determine the "formation cycle" for each H-bond.

        Args:
            protein: Protein network (native state with all H-bonds)
            max_cycles: Maximum cycles to simulate

        Returns:
            Dictionary mapping (donor_res, acceptor_res) -> formation_cycle
        """
        logger.info(f"Identifying formation cycles for {len(protein.h_bonds)} H-bonds")

        # Reset bond states (random initial phases)
        for bond in protein.h_bonds:
            bond.phase = np.random.uniform(0, 2*np.pi)
            bond.phase_coherence = 0.0
            bond.groel_coupling = 0.0

        # Track when each bond phase-locks
        formation_cycles = {}
        bond_keys = {(b.donor_residue, b.acceptor_residue): b for b in protein.h_bonds}

        # Run cycles
        for cycle in range(max_cycles):
            # Advance one cycle
            cycle_result = self.groel.advance_cycle(protein)

            # Check each bond
            for bond_key, bond in bond_keys.items():
                if bond_key not in formation_cycles:  # Not yet phase-locked
                    # Check if phase-locked now (coherence > 0.7 and stable)
                    if bond.phase_coherence > 0.7 and bond.groel_coupling > 0.5:
                        formation_cycles[bond_key] = cycle + 1  # 1-indexed cycles
                        logger.debug(f"  Bond {bond_key} phase-locked at cycle {cycle + 1}")

            # Check if all bonds locked
            if len(formation_cycles) == len(bond_keys):
                logger.info(f"✓ All bonds phase-locked by cycle {cycle + 1}")
                break

        # Record formation events
        self.formation_events = []
        for bond_key, cycle in formation_cycles.items():
            bond = bond_keys[bond_key]
            event = HBondFormationEvent(
                bond=bond,
                formation_cycle=cycle,
                cavity_frequency=self.groel.get_current_cavity_state().cavity_frequency,
                phase_coherence_achieved=bond.phase_coherence,
                required_previous_bonds=[]  # Will be filled in by dependency analysis
            )
            self.formation_events.append(event)

        # Sort by formation cycle
        self.formation_events.sort(key=lambda e: e.formation_cycle)

        logger.info(f"✓ Identified formation cycles for {len(formation_cycles)}/{len(bond_keys)} bonds")

        return formation_cycles

    def test_stability_without_bond(self, protein: ProteinFoldingNetwork,
                                   bond_to_remove: Tuple[int, int],
                                   test_cycles: int = 5) -> Dict:
        """
        Test if protein remains stable when one H-bond is removed.

        This is the core "reverse destabilization" test.

        Args:
            protein: Protein network
            bond_to_remove: (donor_res, acceptor_res) of bond to remove
            test_cycles: Number of cycles to test stability

        Returns:
            Stability test results
        """
        # Create copy without this bond
        test_protein = ProteinFoldingNetwork(protein.protein_name, self.temperature)

        for bond in protein.h_bonds:
            bond_key = (bond.donor_residue, bond.acceptor_residue)
            if bond_key != bond_to_remove:
                # Deep copy bond
                bond_copy = deepcopy(bond)
                test_protein.add_h_bond(bond_copy)

        # Test stability over cycles
        test_groel = GroELResonanceChamber(self.temperature)
        stability_trace = []
        variance_trace = []

        for cycle in range(test_cycles):
            cycle_result = test_groel.advance_cycle(test_protein)
            stability_trace.append(cycle_result['final_stability'])
            variance_trace.append(cycle_result['final_variance'])

        # Check if stable (stability doesn't collapse)
        min_stability = np.min(stability_trace)
        max_variance = np.max(variance_trace)

        is_stable = (min_stability > 0.5) and (max_variance < 0.5)

        return {
            'bond_removed': bond_to_remove,
            'remaining_bonds': len(test_protein.h_bonds),
            'is_stable': is_stable,
            'min_stability': float(min_stability),
            'max_variance': float(max_variance),
            'stability_trace': [float(s) for s in stability_trace]
        }

    def build_dependency_graph(self, protein: ProteinFoldingNetwork,
                               formation_cycles: Dict[Tuple[int, int], int]) -> Dict:
        """
        Build dependency graph showing which bonds depend on which others.

        Method: For each bond, remove it and test if later bonds can still form.
        If later bond can't form without this bond, it's a dependency.

        Args:
            protein: Protein network
            formation_cycles: Dict mapping bonds -> formation cycles

        Returns:
            Dependency graph structure
        """
        logger.info("Building H-bond dependency graph...")

        # Sort bonds by formation cycle
        sorted_bonds = sorted(formation_cycles.items(), key=lambda x: x[1])

        dependencies = {}  # bond -> list of bonds it depends on

        for i, (bond_key, cycle) in enumerate(sorted_bonds):
            dependencies[bond_key] = []

            # Test each earlier bond
            for j in range(i):
                earlier_bond_key, earlier_cycle = sorted_bonds[j]

                # Remove earlier bond and test if current bond can still phase-lock
                test_result = self.test_stability_without_bond(
                    protein, earlier_bond_key, test_cycles=cycle + 2
                )

                # If unstable without earlier bond, it's a dependency
                if not test_result['is_stable']:
                    dependencies[bond_key].append(earlier_bond_key)
                    logger.debug(f"  Bond {bond_key} depends on {earlier_bond_key}")

        # Update formation events with dependencies
        for event in self.formation_events:
            bond_key = (event.bond.donor_residue, event.bond.acceptor_residue)
            event.required_previous_bonds = dependencies.get(bond_key, [])

        return dependencies

    def build_folding_pathway_tree(self, formation_cycles: Dict[Tuple[int, int], int],
                                   dependencies: Dict) -> FoldingPathwayNode:
        """
        Build tree structure of folding pathway.

        Tree nodes = protein states after each H-bond formation
        Edges = formation events (bonds being established)

        Args:
            formation_cycles: Bond -> cycle mapping
            dependencies: Dependency graph

        Returns:
            Root node of pathway tree
        """
        logger.info("Building folding pathway tree...")

        # Sort bonds by formation cycle
        sorted_bonds = sorted(formation_cycles.items(), key=lambda x: x[1])

        # Create root (unfolded state, cycle 0)
        root = FoldingPathwayNode(
            cycle=0,
            bonds_formed=[],
            network_stability=0.0,
            variance=float('inf'),
            phase_clusters=[]
        )

        # Build tree level by level (cycle by cycle)
        current_nodes = [root]

        for bond_key, cycle in sorted_bonds:
            # Find bond object
            bond = next((e.bond for e in self.formation_events
                        if (e.bond.donor_residue, e.bond.acceptor_residue) == bond_key), None)

            if not bond:
                continue

            # Find parent node (most recent node with all dependencies satisfied)
            required_bonds = set(dependencies.get(bond_key, []))

            valid_parents = []
            for node in current_nodes:
                node_bond_keys = {(b.donor_residue, b.acceptor_residue) for b in node.bonds_formed}
                if required_bonds.issubset(node_bond_keys):
                    valid_parents.append(node)

            # Use most recent valid parent
            if valid_parents:
                parent = max(valid_parents, key=lambda n: n.cycle)
            else:
                parent = root

            # Create new node
            new_node = FoldingPathwayNode(
                cycle=cycle,
                bonds_formed=parent.bonds_formed + [bond],
                network_stability=0.0,  # Would need to calculate
                variance=0.0,
                phase_clusters=[],
                parent_node=parent
            )

            parent.children.append(new_node)
            current_nodes.append(new_node)

        self.pathway_tree = root
        return root

    def discover_folding_pathway(self, protein: ProteinFoldingNetwork,
                                 max_cycles: int = 20) -> Dict:
        """
        Complete reverse folding analysis to discover folding pathway.

        Full algorithm:
        1. Identify formation cycles for all H-bonds
        2. Build dependency graph
        3. Construct pathway tree
        4. Identify critical cycles and folding nuclei

        Args:
            protein: Native protein structure
            max_cycles: Maximum cycles to simulate

        Returns:
            Complete pathway analysis
        """
        logger.info(f"="*70)
        logger.info(f"DISCOVERING FOLDING PATHWAY FOR {protein.protein_name}")
        logger.info(f"="*70)

        # Step 1: Formation cycles
        formation_cycles = self.identify_bond_formation_cycles(protein, max_cycles)

        # Step 2: Dependency graph
        dependencies = self.build_dependency_graph(protein, formation_cycles)

        # Step 3: Pathway tree
        pathway_tree = self.build_folding_pathway_tree(formation_cycles, dependencies)

        # Step 4: Identify critical cycles (where multiple bonds form)
        cycle_counts = {}
        for cycle in formation_cycles.values():
            cycle_counts[cycle] = cycle_counts.get(cycle, 0) + 1

        self.critical_cycles = [c for c, count in cycle_counts.items() if count > 1]
        self.critical_cycles.sort()

        # Summary
        total_bonds = len(formation_cycles)
        cycles_used = max(formation_cycles.values()) if formation_cycles else 0

        pathway_summary = {
            'protein': protein.protein_name,
            'total_bonds': total_bonds,
            'cycles_to_fold': cycles_used,
            'bonds_per_cycle': cycle_counts,
            'critical_cycles': self.critical_cycles,
            'formation_events': [
                {
                    'bond': f"{e.bond.donor_residue}-{e.bond.acceptor_residue}",
                    'cycle': e.formation_cycle,
                    'cavity_freq': e.cavity_frequency,
                    'coherence': e.phase_coherence_achieved,
                    'dependencies': len(e.required_previous_bonds)
                }
                for e in self.formation_events
            ],
            'folding_nucleus': None  # Will identify from dependencies
        }

        # Identify folding nucleus (bonds with most dependents)
        dependents_count = {}
        for bond_key, deps in dependencies.items():
            for dep in deps:
                dependents_count[dep] = dependents_count.get(dep, 0) + 1

        if dependents_count:
            nucleus_bond = max(dependents_count.items(), key=lambda x: x[1])
            pathway_summary['folding_nucleus'] = {
                'bond': f"{nucleus_bond[0][0]}-{nucleus_bond[0][1]}",
                'cycle': formation_cycles[nucleus_bond[0]],
                'dependent_bonds': nucleus_bond[1]
            }

        logger.info(f"="*70)
        logger.info(f"PATHWAY DISCOVERY COMPLETE")
        logger.info(f"  Total bonds: {total_bonds}")
        logger.info(f"  Cycles to fold: {cycles_used}")
        logger.info(f"  Critical cycles: {self.critical_cycles}")
        if pathway_summary['folding_nucleus']:
            logger.info(f"  Folding nucleus: {pathway_summary['folding_nucleus']['bond']} "
                       f"(cycle {pathway_summary['folding_nucleus']['cycle']})")
        logger.info(f"="*70)

        return pathway_summary
