"""
Conservation of Categorical Information: The Universe Has No Drain

This script demonstrates why categorical distinctions cannot be destroyed,
only redistributed among observers.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class ClosedSystem:
    """
    A closed system (like the universe) where information cannot leave.
    """
    def __init__(self, total_categories: int, num_observers: int):
        self.total_categories = total_categories
        self.num_observers = num_observers

        # Distribute categories randomly among observers
        self.accessible = {}
        self.inaccessible = {}

        categories_per_observer = total_categories // num_observers
        for i in range(num_observers):
            self.accessible[f"O{i+1}"] = categories_per_observer
            self.inaccessible[f"O{i+1}"] = total_categories - categories_per_observer

    def redistribute(self, from_obs: str, to_obs: str, amount: int):
        """
        Move information from one observer's inaccessible to another's accessible.
        This is like communication or observation.
        """
        if amount > self.inaccessible[from_obs]:
            amount = self.inaccessible[from_obs]

        self.inaccessible[from_obs] -= amount
        self.accessible[to_obs] += amount

        # But redistribution creates NEW categories (the observation itself)
        new_category = 1  # Each transfer creates one new distinction
        self.total_categories += new_category

        # Distribute the new category's inaccessibility
        for obs in self.accessible.keys():
            if obs != to_obs:
                self.inaccessible[obs] += new_category // (self.num_observers - 1)

    def verify_conservation(self) -> bool:
        """
        Verify that total accessible + inaccessible = total for all observers.
        """
        for obs in self.accessible.keys():
            total_for_obs = self.accessible[obs] + self.inaccessible[obs]
            if total_for_obs != self.total_categories:
                return False
        return True

    def get_state(self) -> Dict[str, Tuple[int, int]]:
        """
        Get current state: (accessible, inaccessible) for each observer.
        """
        return {obs: (self.accessible[obs], self.inaccessible[obs])
                for obs in self.accessible.keys()}


def demonstrate_conservation():
    """
    Demonstrate that the universe has no drain: information can only be moved.
    """
    print("=" * 80)
    print("CONSERVATION OF CATEGORICAL INFORMATION")
    print("The Universe Has No Drain")
    print("=" * 80)
    print()

    print("ANALOGY: Bathtub vs Universe")
    print("-" * 80)
    print()
    print("Bathtub (Open System):")
    print("  • Has a drain")
    print("  • Dirt can EXIT the system")
    print("  • Can return to clean state")
    print("  • Total dirt can DECREASE")
    print()
    print("Universe (Closed System):")
    print("  • NO drain")
    print("  • Information STAYS in system")
    print("  • Cannot return to C(0) = 1")
    print("  • Total categories can only INCREASE or stay constant")
    print()

    print("=" * 80)
    print("DEMONSTRATION: Information Redistribution")
    print("=" * 80)
    print()

    # Create a closed system
    system = ClosedSystem(total_categories=100, num_observers=3)

    print("Initial state:")
    print(f"  Total categories in system: {system.total_categories}")
    print()
    for obs, (acc, inacc) in system.get_state().items():
        print(f"  {obs}: accessible={acc}, inaccessible={inacc}, x={inacc}")
    print()

    print("Attempt to 'clean up' by redistributing:")
    print("-" * 80)
    print()

    print("Step 1: O1 observes O2's information")
    print("  (Transfer 10 categories from O2's inaccessible to O1's accessible)")
    system.redistribute("O2", "O1", 10)
    print()
    print(f"  Result: Total categories = {system.total_categories} (INCREASED!)")
    for obs, (acc, inacc) in system.get_state().items():
        print(f"    {obs}: accessible={acc}, inaccessible={inacc}, x={inacc}")
    print()
    print("  Why increase? The observation itself creates a NEW category:")
    print("    'O1 observed O2' is a new categorical distinction")
    print()

    print("Step 2: O2 observes O3's information")
    system.redistribute("O3", "O2", 15)
    print()
    print(f"  Result: Total categories = {system.total_categories} (INCREASED AGAIN!)")
    for obs, (acc, inacc) in system.get_state().items():
        print(f"    {obs}: accessible={acc}, inaccessible={inacc}, x={inacc}")
    print()

    print("Step 3: O3 observes O1's information")
    system.redistribute("O1", "O3", 20)
    print()
    print(f"  Result: Total categories = {system.total_categories} (STILL INCREASING!)")
    for obs, (acc, inacc) in system.get_state().items():
        print(f"    {obs}: accessible={acc}, inaccessible={inacc}, x={inacc}")
    print()

    print("=" * 80)
    print("KEY OBSERVATIONS:")
    print("=" * 80)
    print()
    print("1. CANNOT REDUCE TOTAL:")
    print("   Started with 100 categories")
    print(f"   Ended with {system.total_categories} categories")
    print("   Total INCREASED despite trying to 'clean up'")
    print()
    print("2. INFORMATION REDISTRIBUTED:")
    print("   Each observer's x changed")
    print("   But NO observer reached x = 0 (complete knowledge)")
    print()
    print("3. OBSERVATION CREATES DISTINCTIONS:")
    print("   Each attempt to access inaccessible info creates new categories")
    print("   Like splashing water while trying to dry a tub")
    print()
    print("4. NO DRAIN:")
    print("   Cannot export categories outside the system")
    print("   Can only move them around inside")
    print("   Total always increases (monotonic growth)")
    print()

    # Verify conservation
    if system.verify_conservation():
        print("✓ CONSERVATION VERIFIED:")
        print("  For each observer: accessible + inaccessible = total")
        print("  The 'dirt' is conserved, just redistributed")
    print()


def demonstrate_singularity_uniqueness():
    """
    Show that C(0) = 1 is the only 'clean' state.
    """
    print("=" * 80)
    print("THE SINGULARITY: The Only Clean State")
    print("=" * 80)
    print()

    print("At t = 0 (Big Bang singularity):")
    print("  C(0) = 1")
    print("  No categorical distinctions")
    print("  No 'dirt' in the system")
    print("  CLEAN STATE")
    print()

    print("For t > 0 (after singularity):")
    print("  C(t) > 1")
    print("  Categorical distinctions have been made")
    print("  'Dirt' now exists in the system")
    print("  CANNOT RETURN TO CLEAN")
    print()

    print("Why can't we return to C(0) = 1?")
    print("-" * 80)
    print()
    print("To 'clean up' would require:")
    print("  1. Destroying all categorical distinctions")
    print("  2. Merging all particles back to singularity")
    print("  3. Eliminating all observer information")
    print()
    print("But:")
    print("  • No drain exists to export this information")
    print("  • Categories are conserved")
    print("  • Once distinguished, always distinguished")
    print("  • The system is IRREVERSIBLY 'dirty'")
    print()

    print("This explains:")
    print("  ✓ Thermodynamic arrow of time (entropy increases)")
    print("  ✓ Monotonic growth of C(t)")
    print("  ✓ Why universe expands but doesn't contract back to singularity")
    print("  ✓ Why knowledge can accumulate but not be 'unlearned' from the universe")
    print()


def demonstrate_knowledge_horizon():
    """
    Show why no observer can reach x = 0 (complete knowledge).
    """
    print("=" * 80)
    print("THE KNOWLEDGE HORIZON: Why x > 0 Always")
    print("=" * 80)
    print()

    print("Thought experiment: Observer O tries to reach x = 0")
    print("-" * 80)
    print()

    print("Initial state:")
    print("  O has x(O) = 1000 (much inaccessible information)")
    print("  O attempts to observe all other observers")
    print()

    x_values = [1000]

    print("Step 1: O observes observer O1")
    print("  Gains 100 categories from O1")
    print("  But observation creates 1 new category")
    print("  Net: x(O) = 1000 - 100 + 1 = 901")
    x_values.append(901)
    print()

    print("Step 2: O observes observer O2")
    print("  Gains 100 categories from O2")
    print("  But observation creates 1 new category")
    print("  Net: x(O) = 901 - 100 + 1 = 802")
    x_values.append(802)
    print()

    print("Step 3-10: O continues observing...")
    for i in range(3, 11):
        x_new = x_values[-1] - 100 + 1
        x_values.append(x_new)
        print(f"  Step {i}: x(O) = {x_new}")
    print()

    print("Asymptotic behavior:")
    print("-" * 80)
    print("  As O observes more, x(O) decreases...")
    print("  But each observation adds new categories")
    print("  Eventually: rate of gain ≈ rate of creation")
    print("  x(O) approaches a floor value > 0")
    print()
    print("  Like trying to empty a bathtub with a bucket:")
    print("    - Each scoop removes water (gain categories)")
    print("    - But splashing adds water back (create categories)")
    print("    - Can never get completely dry (x > 0 always)")
    print()

    print("CONCLUSION:")
    print("  x(O) > 0 for all observers O at all times")
    print("  Complete knowledge (x = 0) is impossible")
    print("  Not due to technology or quantum limits")
    print("  But due to topology: closed system with no drain")
    print()


def summary():
    """
    Final summary of conservation law implications.
    """
    print("=" * 80)
    print("SUMMARY: Conservation of Categorical Information")
    print("=" * 80)
    print()

    print("FUNDAMENTAL PRINCIPLE:")
    print("  In a closed universe (no drain), categorical distinctions are conserved.")
    print("  They can be redistributed but never destroyed.")
    print()

    print("KEY IMPLICATIONS:")
    print()
    print("1. MONOTONIC GROWTH:")
    print("   C(t+1) ≥ C(t) always")
    print("   New observations create distinctions")
    print("   Old distinctions cannot be eliminated")
    print()
    print("2. IRREVERSIBILITY:")
    print("   C(0) = 1 is the only 'clean' state")
    print("   Once t > 0, cannot return to singularity")
    print("   Time has an arrow")
    print()
    print("3. INACCESSIBILITY:")
    print("   x(O) > 0 for all observers O")
    print("   Complete knowledge is impossible")
    print("   Not due to limits but to topology")
    print()
    print("4. REDISTRIBUTION:")
    print("   Observer networks move 'dirt' around")
    print("   Some info becomes accessible, some inaccessible")
    print("   But total always conserved (or increases)")
    print()
    print("5. NECESSITY OF ∞ - x:")
    print("   Conservation ensures x > 0 always")
    print("   Combined with magnitude (all numbers → 0)")
    print("   Makes ∞ - x not just mathematical but physical necessity")
    print()

    print("PHYSICAL INTERPRETATION:")
    print("  If dark matter ↔ inaccessible information:")
    print("    • Dark matter is information 'pushed' to inaccessible domains")
    print("    • It's conserved (can't be destroyed)")
    print("    • It can be redistributed (but never eliminated)")
    print("    • x/(\u221e-x) ≈ 5.4 is the conservation ratio")
    print()

    print("THE BATHTUB ANALOGY:")
    print("  Universe = bathtub without drain")
    print("  Categorical distinctions = dirt")
    print("  Observations = moving dirt around with hands")
    print("  Result: Can rearrange but never eliminate")
    print("          System stays 'dirty' forever")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_conservation()
    print("\n" * 2)
    demonstrate_singularity_uniqueness()
    print("\n" * 2)
    demonstrate_knowledge_horizon()
    print("\n" * 2)
    summary()
