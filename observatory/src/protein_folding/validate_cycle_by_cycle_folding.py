"""
Validation Script: Cycle-by-Cycle Phase-Locked Protein Folding.

Tests the complete phase-locking framework based on papers:
- Proton Maxwell Demons with phase dynamics
- GroEL as cyclic resonance chamber
- Reverse folding with cycle tracking
- O₂ master clock synchronization
"""

import sys
from pathlib import Path
import logging
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from proton_maxwell_demon import (
    ProtonMaxwellDemon,
    create_h_bond_oscillator,
    O2_MASTER_CLOCK_HZ,
    GROEL_BASE_HZ
)
from protein_folding_network import ProteinFoldingNetwork
from groel_resonance_chamber import GroELResonanceChamber
from reverse_folding_algorithm import ReverseFoldingAlgorithm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_1_phase_locked_proton_demon():
    """Test 1: Proton demon with phase-locking capability."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: PHASE-LOCKED PROTON DEMON")
    logger.info("="*70)

    demon = ProtonMaxwellDemon(temperature=310.0)

    # Create test H-bond
    bond = create_h_bond_oscillator(
        donor='N', acceptor='O',
        donor_res=10, acceptor_res=25,
        length=2.8, angle=175.0,
        temperature=310.0
    )

    logger.info(f"Created H-bond: residues {bond.donor_residue}-{bond.acceptor_residue}")
    logger.info(f"  Frequency: {bond.frequency:.2e} Hz")
    logger.info(f"  Initial phase: {bond.phase:.2f} rad")

    # Test phase-locking to GroEL cavity
    groel_freq = GROEL_BASE_HZ * 5  # 5 Hz cavity frequency
    groel_phase = np.pi / 2

    logger.info(f"\nTesting phase-lock to GroEL cavity:")
    logger.info(f"  GroEL frequency: {groel_freq:.2f} Hz")

    # Calculate phase-lock strength
    lock_strength = bond.calculate_phase_lock_strength(groel_freq)
    logger.info(f"  Phase-lock strength: {lock_strength:.3f}")

    # Simulate phase evolution
    for t in [0.001, 0.01, 0.1]:
        bond.update_phase(t, groel_phase, groel_freq)
        logger.info(f"  After {t}s: phase={bond.phase:.2f} rad, coherence={bond.phase_coherence:.3f}")

    assert bond.frequency > 1e13, "Proton frequency should be > 10^13 Hz"
    assert 0 <= bond.phase_coherence <= 1.0, "Coherence must be in [0,1]"

    logger.info("✓ Phase-locked proton demon working correctly")

    return {
        'frequency': bond.frequency,
        'phase_lock_strength': lock_strength,
        'final_coherence': bond.phase_coherence
    }


def test_2_protein_network_phase_coherence():
    """Test 2: Protein network with phase-coherence tracking."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: PROTEIN NETWORK PHASE-COHERENCE")
    logger.info("="*70)

    protein = ProteinFoldingNetwork("test_protein", temperature=310.0)

    # Create small H-bond network (beta sheet)
    bonds = [
        (5, 25, 2.9, 178),
        (7, 23, 2.8, 175),
        (9, 21, 2.7, 172),
        (11, 19, 2.8, 176),
    ]

    for i, (d_res, a_res, length, angle) in enumerate(bonds):
        bond = create_h_bond_oscillator(
            donor='N', acceptor='O',
            donor_res=d_res, acceptor_res=a_res,
            length=length, angle=angle,
            temperature=310.0
        )
        protein.add_h_bond(bond)

    logger.info(f"Created protein network with {len(protein.h_bonds)} H-bonds")

    # Set GroEL state
    groel_freq = GROEL_BASE_HZ * 3
    groel_phase = 0.0
    protein.update_groel_state(cycle=1, phase=groel_phase, frequency=groel_freq)

    # Calculate network properties
    stability = protein.calculate_network_stability()
    variance = protein.calculate_network_variance()

    logger.info(f"\nNetwork properties:")
    logger.info(f"  Stability: {stability:.3f}")
    logger.info(f"  Variance: {variance:.3f}")
    logger.info(f"  Mean coherence: {protein.network_coherence:.3f}")

    # Find phase-coherence clusters
    clusters = protein.find_phase_coherence_clusters(coherence_threshold=0.5)
    logger.info(f"\nPhase-coherence clusters: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        logger.info(f"  Cluster {i+1}: {len(cluster.bonds)} bonds, "
                   f"coherence={cluster.coherence:.3f}, "
                   f"center_phase={cluster.center_phase:.2f} rad")

    # Identify folding nucleus
    nucleus = protein.identify_folding_nucleus()
    if nucleus:
        logger.info(f"\nFolding nucleus: {len(nucleus.bonds)} bonds, coherence={nucleus.coherence:.3f}")

    assert len(protein.h_bonds) == 4, "Should have 4 H-bonds"
    assert stability >= 0.0, "Stability should be non-negative"

    logger.info("✓ Protein network phase-coherence working correctly")

    return {
        'total_bonds': len(protein.h_bonds),
        'stability': stability,
        'variance': variance,
        'num_clusters': len(clusters),
        'nucleus_size': len(nucleus.bonds) if nucleus else 0
    }


def test_3_groel_cyclic_resonance():
    """Test 3: GroEL cyclic resonance chamber."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: GROEL CYCLIC RESONANCE CHAMBER")
    logger.info("="*70)

    groel = GroELResonanceChamber(temperature=310.0)

    # Create test protein
    protein = ProteinFoldingNetwork("test_cyclic", temperature=310.0)

    # Add H-bonds
    for i in range(6):
        bond = create_h_bond_oscillator(
            donor='N', acceptor='O',
            donor_res=i*5, acceptor_res=i*5+15,
            length=2.8, angle=175.0,
            temperature=310.0
        )
        protein.add_h_bond(bond)

    logger.info(f"Testing GroEL cycles with {len(protein.h_bonds)}-bond protein")

    # Run 5 ATP cycles
    n_cycles = 5
    logger.info(f"\nRunning {n_cycles} ATP cycles...")

    for cycle in range(n_cycles):
        # Get cavity state
        cavity_state = groel.get_current_cavity_state()
        logger.info(f"\nCycle {cycle + 1}:")
        logger.info(f"  ATP state: {cavity_state.atp_state}")
        logger.info(f"  Cavity freq: {cavity_state.cavity_frequency:.2f} Hz")
        logger.info(f"  Cavity volume: {cavity_state.cavity_volume:.0f} Ų")

        # Advance cycle
        cycle_result = groel.advance_cycle(protein)
        logger.info(f"  Final stability: {cycle_result['final_stability']:.3f}")
        logger.info(f"  Final variance: {cycle_result['final_variance']:.3f}")

    # Check results
    assert len(groel.protein_history) == n_cycles, f"Should have {n_cycles} cycle records"
    assert groel.best_cycle is not None, "Should have identified best cycle"

    logger.info(f"\nBest cycle: {groel.best_cycle} (stability={groel.best_stability:.3f})")
    logger.info("✓ GroEL cyclic resonance working correctly")

    return {
        'cycles_run': n_cycles,
        'best_cycle': groel.best_cycle,
        'best_stability': groel.best_stability,
        'final_variance': groel.protein_history[-1]['final_variance']
    }


def test_4_complete_folding_simulation():
    """Test 4: Complete folding simulation."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: COMPLETE FOLDING SIMULATION")
    logger.info("="*70)

    groel = GroELResonanceChamber(temperature=310.0)
    protein = ProteinFoldingNetwork("ubiquitin_model", temperature=310.0)

    # Create small protein model (10 H-bonds)
    logger.info("Creating 10-bond protein model...")
    for i in range(10):
        bond = create_h_bond_oscillator(
            donor='N', acceptor='O',
            donor_res=i*3, acceptor_res=i*3+10,
            length=2.7 + np.random.uniform(-0.2, 0.2),
            angle=175 + np.random.uniform(-5, 5),
            temperature=310.0
        )
        protein.add_h_bond(bond)

    # Run folding simulation
    result = groel.run_folding_simulation(
        protein,
        max_cycles=15,
        stability_threshold=0.7
    )

    logger.info(f"\nFolding simulation results:")
    logger.info(f"  Cycles run: {result['cycles_run']}")
    logger.info(f"  Folding complete: {result['folding_complete']}")
    logger.info(f"  Best cycle: {result['best_cycle']}")
    logger.info(f"  Final stability: {result['final_stability']:.3f}")
    logger.info(f"  Final variance: {result['final_variance']:.3f}")

    # Get folding pathway
    pathway = groel.identify_folding_pathway(protein)
    logger.info(f"\nFolding pathway events: {len(pathway)}")
    for event in pathway[:5]:  # Show first 5
        logger.info(f"  Cycle {event['cycle']}: {event['event']} "
                   f"(stability={event['stability']:.3f})")

    assert result['cycles_run'] > 0, "Should have run at least one cycle"
    assert result['final_stability'] > 0, "Should have positive stability"

    logger.info("✓ Complete folding simulation working correctly")

    return result


def test_5_reverse_folding_pathway_discovery():
    """Test 5: Reverse folding pathway discovery (cycle-by-cycle)."""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: REVERSE FOLDING PATHWAY DISCOVERY")
    logger.info("="*70)

    algorithm = ReverseFoldingAlgorithm(temperature=310.0)
    protein = ProteinFoldingNetwork("test_pathway", temperature=310.0)

    # Create small protein (8 H-bonds in beta-sheet-like structure)
    logger.info("Creating 8-bond protein model...")
    bond_specs = [
        (5, 25, 2.8, 175),
        (7, 23, 2.9, 173),
        (9, 21, 2.7, 177),
        (11, 19, 2.8, 176),
        (13, 17, 2.9, 174),
        (15, 35, 2.7, 178),
        (17, 33, 2.8, 175),
        (19, 31, 2.9, 173),
    ]

    for d_res, a_res, length, angle in bond_specs:
        bond = create_h_bond_oscillator(
            donor='N', acceptor='O',
            donor_res=d_res, acceptor_res=a_res,
            length=length, angle=angle,
            temperature=310.0
        )
        protein.add_h_bond(bond)

    logger.info(f"Running reverse folding analysis...")

    # Discover pathway (more cycles for THz phase-locking)
    pathway = algorithm.discover_folding_pathway(protein, max_cycles=30)

    logger.info(f"\nPathway discovery results:")
    logger.info(f"  Total bonds: {pathway['total_bonds']}")
    logger.info(f"  Cycles to fold: {pathway['cycles_to_fold']}")
    logger.info(f"  Critical cycles: {pathway['critical_cycles']}")

    logger.info(f"\nBonds per cycle:")
    for cycle, count in sorted(pathway['bonds_per_cycle'].items()):
        logger.info(f"  Cycle {cycle}: {count} bonds")

    if pathway['folding_nucleus']:
        nucleus = pathway['folding_nucleus']
        logger.info(f"\nFolding nucleus:")
        logger.info(f"  Bond: {nucleus['bond']}")
        logger.info(f"  Cycle: {nucleus['cycle']}")
        logger.info(f"  Dependent bonds: {nucleus['dependent_bonds']}")

    logger.info(f"\nFormation events (first 5):")
    for event in pathway['formation_events'][:5]:
        logger.info(f"  Cycle {event['cycle']}: Bond {event['bond']} "
                   f"(coherence={event['coherence']:.3f}, deps={event['dependencies']})")

    assert pathway['total_bonds'] == 8, "Should track all 8 bonds"
    assert pathway['cycles_to_fold'] > 0, "Should require at least one cycle"

    logger.info("✓ Reverse folding pathway discovery working correctly")

    return pathway


def main():
    """Run all validation tests."""
    logger.info("\n" + "#"*70)
    logger.info("# CYCLE-BY-CYCLE PHASE-LOCKED PROTEIN FOLDING VALIDATION")
    logger.info("# Based on cytoplasmic phase-locking papers")
    logger.info("#"*70)

    results = {}

    try:
        results['test_1'] = test_1_phase_locked_proton_demon()
        results['test_2'] = test_2_protein_network_phase_coherence()
        results['test_3'] = test_3_groel_cyclic_resonance()
        results['test_4'] = test_4_complete_folding_simulation()
        results['test_5'] = test_5_reverse_folding_pathway_discovery()

        # Save results
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / 'cycle_by_cycle_validation.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("\n" + "#"*70)
        logger.info("# ALL TESTS PASSED")
        logger.info(f"# Results saved to: {output_file}")
        logger.info("#"*70)

    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
