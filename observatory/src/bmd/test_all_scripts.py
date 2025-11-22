"""
test_all_scripts.py

Quick test to verify all BMD validation scripts are working.
Tests imports and basic functionality.
"""

import sys
import traceback

def test_script(script_name, test_func):
    """Test a single script"""
    print(f"\nTesting {script_name}...")
    print("-" * 60)
    try:
        test_func()
        print(f"✓ {script_name} PASSED")
        return True
    except Exception as e:
        print(f"✗ {script_name} FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_categorical_tracker():
    """Test categorical_tracker imports and basic functionality"""
    from categorical_tracker import (
        CategoricalState,
        EquivalenceClass,
        CategoricalTracker
    )

    # Create tracker
    tracker = CategoricalTracker(observable_precision=0.01)

    # Create mock equivalence class
    equiv = EquivalenceClass(
        class_id=0,
        representative_state=0,
        observable_signature=(1.0, 1.0, 1.0, 1.0)
    )
    equiv.member_states.add(0)

    assert equiv.degeneracy == 1
    assert equiv.information_content >= 0
    print("  - CategoricalTracker initialized")
    print("  - EquivalenceClass working")

def test_recursive_bmd():
    """Test recursive_bmd_analysis imports and basic functionality"""
    from recursive_bmd_analysis import (
        BMDLevel,
        RecursiveBMDAnalyzer
    )

    # Create analyzer
    analyzer = RecursiveBMDAnalyzer(max_depth=2)

    # Build small hierarchy
    S_global = (1.0, 2.0, 3.0)
    context = {
        'global_equiv_class_size': 1000,
        'knowledge_uncertainty': 0.3,
        'knowledge_progress': 0.5,
        'knowledge_constraints': 0.7,
        'time_uncertainty': 0.2,
        'time_progress': 0.8,
        'time_constraints': 0.9,
        'entropy_knowledge': 0.4,
        'entropy_progress': 0.6,
        'entropy_constraints': 0.8
    }

    analyzer.build_recursive_hierarchy(S_global, context)

    # Verify hierarchy
    assert len(analyzer.hierarchy[0]) == 1
    assert len(analyzer.hierarchy[1]) == 3
    assert len(analyzer.hierarchy[2]) == 9

    print("  - RecursiveBMDAnalyzer initialized")
    print("  - Hierarchy built correctly")
    print(f"  - Total BMDs: {sum(len(analyzer.hierarchy[i]) for i in range(3))}")

def test_mechanics():
    """Test mechanics (PrisonerSystem) imports"""
    from mechanics import (
        Particle,
        Compartment,
        MaxwellDemon,
        PrisonerSystem,
        ParticleState
    )

    # Create small system
    system = PrisonerSystem(
        n_particles=10,
        demon_params={
            'information_capacity': 5.0,
            'selection_threshold': 1.0,
            'error_rate': 0.1,
            'memory_cost': 0.01
        }
    )

    # Run a few steps
    for _ in range(5):
        system.step()

    print("  - PrisonerSystem initialized")
    print(f"  - Simulated {system.time:.3f} time units")
    print(f"  - Demon accuracy: {system.demon.accuracy:.2%}")

def test_thermodynamics():
    """Test thermodynamics imports"""
    from thermodynamics import (
        ThermodynamicState,
        ThermodynamicsAnalyzer
    )

    analyzer = ThermodynamicsAnalyzer()

    # Create mock state
    state = ThermodynamicState(
        temperature_A=1.0,
        temperature_B=1.5,
        entropy_A=1.0,
        entropy_B=1.2,
        demon_entropy_cost=0.1,
        particles_A=50,
        particles_B=50
    )

    assert state.total_entropy > 0
    assert state.temperature_gradient != 0

    print("  - ThermodynamicsAnalyzer initialized")
    print(f"  - Total entropy: {state.total_entropy:.3f}")

def test_main_simulation():
    """Test main_simulation imports"""
    from main_simulation import run_simulation

    # Run tiny simulation
    system, analyzer = run_simulation(
        n_particles=10,
        n_steps=10,
        demon_params={
            'information_capacity': 5.0,
            'selection_threshold': 1.0,
            'error_rate': 0.1,
            'memory_cost': 0.01
        },
        verbose=False
    )

    print("  - run_simulation executed")
    print(f"  - Final temp difference: {system.temperature_difference:.3f}")

def main():
    """Run all tests"""
    print("=" * 70)
    print("BMD VALIDATION FRAMEWORK - SCRIPT VERIFICATION")
    print("=" * 70)

    tests = [
        ("categorical_tracker.py", test_categorical_tracker),
        ("recursive_bmd_analysis.py", test_recursive_bmd),
        ("mechanics.py", test_mechanics),
        ("thermodynamics.py", test_thermodynamics),
        ("main_simulation.py", test_main_simulation),
    ]

    results = []
    for script_name, test_func in tests:
        passed = test_script(script_name, test_func)
        results.append((script_name, passed))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for script_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {script_name}")

    print()
    print(f"Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✓✓✓ ALL SCRIPTS WORKING ✓✓✓")
        print("\nYou can now run:")
        print("  python recursive_bmd_analysis.py")
        print("  python validate_st_stellas.py")
        print("  python experiments.py")
    else:
        print("\n⚠ Some scripts have issues - check errors above")

    print("=" * 70)

if __name__ == "__main__":
    main()
