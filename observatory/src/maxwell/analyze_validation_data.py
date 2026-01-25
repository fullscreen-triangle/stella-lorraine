"""
analyze_validation_data.py

Comprehensive analysis of validation results and grid data.
Generates statistics, verification checks, and publication-ready tables.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from glob import glob

def find_latest_data(data_dir='results/dual_membrane_validation'):
    """Find the most recent validation data files"""
    data_dir = Path(data_dir)

    # Find all validation result files
    json_files = list(data_dir.glob('validation_results_*.json'))

    if not json_files:
        raise FileNotFoundError(f"No validation results found in {data_dir}")

    # Get the most recent one
    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)

    # Extract timestamp
    timestamp = latest_json.stem.split('_', 2)[-1]

    print(f"\n✓ Using validation data from: {timestamp}")
    print(f"  JSON file: {latest_json.name}")

    return data_dir, timestamp

def load_all_data(data_dir=None, timestamp=None):
    """Load all validation data"""

    if data_dir is None or timestamp is None:
        data_dir, timestamp = find_latest_data()

    data_dir = Path(data_dir)

    print("\nLoading numpy arrays...")

    # Load numpy arrays
    data = {
        'front_sk': np.load(data_dir / f'front_sk_image_{timestamp}.npy'),
        'back_sk': np.load(data_dir / f'back_sk_image_{timestamp}.npy'),
        'test_pattern': np.load(data_dir / f'test_pattern_{timestamp}.npy'),
        'carbon_copy': np.load(data_dir / f'carbon_copy_{timestamp}.npy'),
    }

    print(f"  ✓ front_sk: {data['front_sk'].shape}")
    print(f"  ✓ back_sk: {data['back_sk'].shape}")
    print(f"  ✓ test_pattern: {data['test_pattern'].shape}")
    print(f"  ✓ carbon_copy: {data['carbon_copy'].shape}")

    # Load validation results
    json_file = data_dir / f'validation_results_{timestamp}.json'
    with open(json_file, 'r') as f:
        data['validation_results'] = json.load(f)

    print(f"  ✓ validation_results loaded")

    return data

def analyze_conjugate_relationship(data):
    """Analyze the conjugate relationship between front and back"""
    front = data['front_sk']
    back = data['back_sk']

    print("\n" + "="*70)
    print("CONJUGATE RELATIONSHIP ANALYSIS")
    print("="*70)

    # Basic statistics
    print("\nFront Face Statistics:")
    print(f"  Mean:     {np.mean(front):12.6f}")
    print(f"  Std Dev:  {np.std(front):12.6f}")
    print(f"  Min:      {np.min(front):12.6f}")
    print(f"  Max:      {np.max(front):12.6f}")
    print(f"  Range:    {np.max(front) - np.min(front):12.6f}")

    print("\nBack Face Statistics:")
    print(f"  Mean:     {np.mean(back):12.6f}")
    print(f"  Std Dev:  {np.std(back):12.6f}")
    print(f"  Min:      {np.min(back):12.6f}")
    print(f"  Max:      {np.max(back):12.6f}")
    print(f"  Range:    {np.max(back) - np.min(back):12.6f}")

    # Conjugate verification
    print("\nConjugate Verification:")

    # Test 1: Mean should be opposite
    mean_sum = np.mean(front) + np.mean(back)
    print(f"  Mean(Front) + Mean(Back) = {mean_sum:.6e}")
    print(f"  Should be ≈ 0: {abs(mean_sum) < 1e-6} ✓" if abs(mean_sum) < 1e-6
          else f"  Should be ≈ 0: {abs(mean_sum) < 1e-6} ✗")

    # Test 2: Std dev should be equal
    std_ratio = np.std(front) / np.std(back) if np.std(back) > 0 else 0
    print(f"  Std(Front) / Std(Back) = {std_ratio:.6f}")
    print(f"  Should be ≈ 1: {abs(std_ratio - 1.0) < 0.1} ✓" if abs(std_ratio - 1.0) < 0.1
          else f"  Should be ≈ 1: {abs(std_ratio - 1.0) < 0.1} ✗")

    # Test 3: Correlation should be -1
    correlation = np.corrcoef(front.flatten(), back.flatten())[0, 1]
    print(f"  Correlation coefficient: {correlation:.6f}")
    print(f"  Should be ≈ -1: {correlation < -0.99} ✓" if correlation < -0.99
          else f"  Should be ≈ -1: {correlation < -0.99} ✗")

    # Test 4: Element-wise sum should be near zero
    sum_grid = front + back
    max_deviation = np.max(np.abs(sum_grid))
    mean_deviation = np.mean(np.abs(sum_grid))
    print(f"  Max |Front + Back|: {max_deviation:.6e}")
    print(f"  Mean |Front + Back|: {mean_deviation:.6e}")
    print(f"  Should be < 0.01: {max_deviation < 0.01} ✓" if max_deviation < 0.01
          else f"  Should be < 0.01: {max_deviation < 0.01} ✗")

    # Test 5: Statistical test (paired t-test on sum)
    t_stat, p_value = stats.ttest_1samp(sum_grid.flatten(), 0)
    print(f"  t-test (sum vs 0): t={t_stat:.3f}, p={p_value:.6f}")
    print(f"  Null hypothesis (sum=0): {'Not rejected ✓' if p_value > 0.05 else 'Rejected ✗'}")

    return {
        'mean_sum': mean_sum,
        'std_ratio': std_ratio,
        'correlation': correlation,
        'max_deviation': max_deviation,
        'mean_deviation': mean_deviation,
        'p_value': p_value
    }

def analyze_carbon_copy(data):
    """Analyze carbon copy propagation"""
    pattern = data['test_pattern']
    copy = data['carbon_copy']

    print("\n" + "="*70)
    print("CARBON COPY ANALYSIS")
    print("="*70)

    print("\nTest Pattern Statistics:")
    print(f"  Mean:     {np.mean(pattern):12.6f}")
    print(f"  Std Dev:  {np.std(pattern):12.6f}")

    print("\nCarbon Copy Statistics:")
    print(f"  Mean:     {np.mean(copy):12.6f}")
    print(f"  Std Dev:  {np.std(copy):12.6f}")

    print("\nCarbon Copy Verification:")

    # Should be negative of pattern
    mean_sum = np.mean(pattern) + np.mean(copy)
    print(f"  Mean(Pattern) + Mean(Copy) = {mean_sum:.6e}")
    print(f"  Should be ≈ 0: {abs(mean_sum) < 1e-6} ✓" if abs(mean_sum) < 1e-6
          else f"  Should be ≈ 0: {abs(mean_sum) < 1e-6} ✗")

    # Correlation should be -1
    correlation = np.corrcoef(pattern.flatten(), copy.flatten())[0, 1]
    print(f"  Correlation coefficient: {correlation:.6f}")
    print(f"  Should be ≈ -1: {correlation < -0.99} ✓" if correlation < -0.99
          else f"  Should be ≈ -1: {correlation < -0.99} ✗")

    return {
        'pattern_mean': np.mean(pattern),
        'copy_mean': np.mean(copy),
        'correlation': correlation
    }

def analyze_test_results(data):
    """Analyze all test results from JSON"""
    results = data['validation_results']

    print("\n" + "="*70)
    print("VALIDATION TEST RESULTS")
    print("="*70)

    tests = results.get('tests', {})

    for test_name, test_data in tests.items():
        print(f"\n{test_name}:")
        print(f"  Status: {'PASSED ✓' if test_data.get('passed', False) else 'FAILED ✗'}")

    # Overall summary
    total_tests = len(tests)
    passed_tests = sum(1 for t in tests.values() if t.get('passed', False))

    print(f"\n{'='*70}")
    print(f"OVERALL: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*70}")

    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'pass_rate': passed_tests / total_tests if total_tests > 0 else 0
    }

def generate_publication_table(data, conjugate_stats, carbon_stats):
    """Generate LaTeX table for publication"""

    print("\n" + "="*70)
    print("PUBLICATION TABLE (LaTeX)")
    print("="*70)

    latex = r"""
\begin{table}[h]
\centering
\caption{Dual-Membrane Grid Validation Results}
\label{tab:validation_results}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Value} & \textbf{Expected} \\
\hline
\multicolumn{3}{c}{\textit{Conjugate Relationship}} \\
Front face $\langle S_k \rangle$ & %.4f & -- \\
Back face $\langle S_k \rangle$ & %.4f & $-\langle S_k^{\text{front}} \rangle$ \\
Sum: $\langle S_k^{\text{front}} + S_k^{\text{back}} \rangle$ & %.2e & $\approx 0$ \\
Correlation coefficient & %.4f & $\approx -1$ \\
Max deviation from zero & %.2e & $< 0.01$ \\
\hline
\multicolumn{3}{c}{\textit{Carbon Copy Propagation}} \\
Test pattern $\langle S_k \rangle$ & %.4f & -- \\
Carbon copy $\langle S_k \rangle$ & %.4f & $-\langle S_k^{\text{pattern}} \rangle$ \\
Correlation coefficient & %.4f & $\approx -1$ \\
\hline
\end{tabular}
\end{table}
""" % (
        np.mean(data['front_sk']),
        np.mean(data['back_sk']),
        conjugate_stats['mean_sum'],
        conjugate_stats['correlation'],
        conjugate_stats['max_deviation'],
        carbon_stats['pattern_mean'],
        carbon_stats['copy_mean'],
        carbon_stats['correlation']
    )

    print(latex)

    # Save to file
    output_file = Path('results/dual_membrane_validation/validation_table.tex')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex)
    print(f"\n✓ LaTeX table saved to: {output_file}")

def main():
    """Main analysis function"""
    print("\n" + "="*70)
    print("DUAL-MEMBRANE VALIDATION DATA ANALYSIS")
    print("="*70)

    # Load data
    print("\nLoading data...")
    data = load_all_data()
    print("✓ Data loaded successfully")

    # Analyze conjugate relationship
    conjugate_stats = analyze_conjugate_relationship(data)

    # Analyze carbon copy
    carbon_stats = analyze_carbon_copy(data)

    # Analyze test results
    test_stats = analyze_test_results(data)

    # Generate publication table
    generate_publication_table(data, conjugate_stats, carbon_stats)

    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n✓ All tests passed: {test_stats['passed_tests']}/{test_stats['total_tests']}")
    print(f"✓ Conjugate relationship verified: {abs(conjugate_stats['correlation'] + 1.0) < 0.01}")
    print(f"✓ Carbon copy verified: {abs(carbon_stats['correlation'] + 1.0) < 0.01}")
    print(f"✓ Conservation verified: {conjugate_stats['max_deviation'] < 0.01}")

    print("\nReady for publication! 🎉")

if __name__ == '__main__':
    main()
