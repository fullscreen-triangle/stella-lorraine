#!/usr/bin/env python3
"""
Stella-Lorraine Quick Start Demo

Run this script to test the validation framework.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from precision.experiment import run_quick_validation_test

    print("🚀 Running Stella-Lorraine validation test...")
    print("This may take a few seconds...")

    # Run a quick semantic distance validation test
    result = run_quick_validation_test("semantic_distance")

    print(f"\n📊 Results:")
    print(f"   Experiment: {result.experiment_id}")
    print(f"   Success: {'✅' if result.success else '❌'} {result.success}")
    print(f"   Duration: {result.duration:.2f}s")
    print(f"   Overall Confidence: {result.get_overall_confidence():.4f}")
    print(f"   Precision Improvement: {result.get_precision_improvement_validated():.2f}×")

    if result.error_messages:
        print(f"   Errors: {result.error_messages}")

    print("\n🎉 Stella-Lorraine framework is working!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure the installation completed successfully.")
except Exception as e:
    print(f"❌ Test error: {e}")
    print("There may be an issue with the framework setup.")

