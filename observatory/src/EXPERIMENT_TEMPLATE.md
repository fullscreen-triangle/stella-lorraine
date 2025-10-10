# Scientific Experiment Template

## Standard Structure for All Experiment Scripts

Each experiment script should follow this structure for publication-ready results:

### 1. File Header
```python
#!/usr/bin/env python3
"""
Module Title
============
Brief description of what this experiment measures

Scientific Method:
- Hypothesis: What we're testing
- Method: How we measure it
- Expected Results: What we predict
"""
```

### 2. Required Imports
```python
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
```

### 3. Main Experiment Function
```python
def main():
    """
    Main experimental function

    Workflow:
    1. Setup (timestamp, directories, logging)
    2. Configuration (parameters, random seeds)
    3. Execution (run experiment)
    4. Analysis (process results)
    5. Saving (JSON results + PNG figures)

    Returns:
        results_dict, figure_path
    """
    # [1/5] Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                               'results', 'experiment_name')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("   EXPERIMENT: [NAME]")
    print("   [DESCRIPTION]")
    print("=" * 70)
    print(f"\n   Timestamp: {timestamp}")
    print(f"   Results directory: {results_dir}")

    # [2/5] Configuration
    np.random.seed(42)  # Reproducibility!

    config = {
        'parameter1': value1,
        'parameter2': value2,
        # ... all experimental parameters
    }

    print(f"\n📊 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # [3/5] Execute Experiment
    print(f"\n[3/5] Running experiment...")
    results = run_experiment(config)

    # [4/5] Analysis
    print(f"\n[4/5] Analyzing results...")
    analysis = analyze_results(results)

    print(f"\n" + "=" * 70)
    print(f"   RESULTS")
    print(f"=" * 70)
    # Print key findings

    # [5/5] Save Results
    print(f"\n[5/5] Saving results and generating visualizations...")

    # Prepare JSON-serializable results
    results_to_save = {
        'timestamp': timestamp,
        'experiment': 'experiment_name',
        'configuration': config,
        'results': {
            # Convert numpy/complex types to Python types
            'metric1': float(analysis['metric1']),
            'metric2': [float(x) for x in analysis['metric2']],
            # ...
        },
        'metadata': {
            'version': '1.0',
            'author': 'Stella-Lorraine Observatory',
            'reproducible': True
        }
    }

    # Save JSON
    results_file = os.path.join(results_dir, f'results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"   ✓ Results saved: {results_file}")

    # Generate visualizations
    fig = create_publication_figure(results, analysis, config)

    # Save figure
    figure_file = os.path.join(results_dir, f'figure_{timestamp}.png')
    plt.savefig(figure_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Figure saved: {figure_file}")

    plt.show()

    print(f"\n✨ Experiment complete!")
    print(f"   Results: {results_file}")
    print(f"   Figure:  {figure_file}")

    return results_to_save, figure_file
```

### 4. Visualization Function
```python
def create_publication_figure(results, analysis, config):
    """
    Create publication-quality figure with 6 panels

    Standard layout:
    [Panel 1] [Panel 2] [Panel 3]
    [Panel 4] [Panel 5] [Panel 6]

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Main result
    ax1 = plt.subplot(2, 3, 1)
    # Plot main finding
    ax1.set_xlabel('X Label', fontsize=12)
    ax1.set_ylabel('Y Label', fontsize=12)
    ax1.set_title('Panel 1: Main Result', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Panel 2-5: Supporting analysis
    # ...

    # Panel 6: Summary statistics (text box)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
    EXPERIMENT SUMMARY

    Configuration:
    • Param1: {config['parameter1']}
    • Param2: {config['parameter2']}

    Key Results:
    • Metric1: {analysis['metric1']:.2e}
    • Metric2: {analysis['metric2']:.2e}

    Status: ✓ Success
    """

    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Experiment Title', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig
```

### 5. Entry Point
```python
if __name__ == "__main__":
    results, figure = main()
```

---

## Checklist for Each Experiment Script

- [ ] Timestamp for reproducibility
- [ ] Random seed set (np.random.seed(42))
- [ ] All parameters documented in config dict
- [ ] Results saved as JSON
- [ ] Figure saved as PNG (300 DPI)
- [ ] 6-panel visualization layout
- [ ] Progress indicators ([1/5], [2/5], etc.)
- [ ] Summary statistics in Panel 6
- [ ] Proper error handling
- [ ] Docstrings for all functions
- [ ] Type hints where applicable

---

## Directory Structure
```
results/
├── experiment_name/
│   ├── results_20251010_120000.json
│   ├── figure_20251010_120000.png
│   ├── results_20251010_120100.json
│   └── figure_20251010_120100.png
├── another_experiment/
│   └── ...
```

---

## JSON Results Format
```json
{
  "timestamp": "20251010_120000",
  "experiment": "experiment_name",
  "configuration": {
    "parameter1": value1,
    "parameter2": value2
  },
  "results": {
    "metric1": 1.23e-15,
    "metric2": [1.0, 2.0, 3.0],
    "analysis": {
      "precision": 47e-21,
      "enhancement": 1e57
    }
  },
  "metadata": {
    "version": "1.0",
    "reproducible": true,
    "random_seed": 42
  }
}
```

---

## Figure Quality Standards

1. **Resolution**: 300 DPI minimum
2. **Format**: PNG for publication
3. **Size**: 16×12 inches (6-panel layout)
4. **Fonts**: 12pt labels, 14pt titles, 16pt suptitle
5. **Colors**: Distinct, colorblind-friendly palette
6. **Grid**: Alpha=0.3 for visibility without clutter
7. **Legend**: Always labeled where applicable
8. **Axes**: Properly labeled with units

---

## Publication-Ready Output

Each experiment produces:
1. **JSON file**: Machine-readable results for analysis
2. **PNG figure**: Publication-quality visualization
3. **Console output**: Human-readable summary
4. **Timestamps**: Full reproducibility trail

This ensures every experiment is:
- ✅ Reproducible
- ✅ Documented
- ✅ Publication-ready
- ✅ Scientifically rigorous
