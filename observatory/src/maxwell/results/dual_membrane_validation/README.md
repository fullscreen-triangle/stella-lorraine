# Dual-Membrane Pixel Demon Validation Results

This directory contains validation results for the Dual-Membrane Pixel Maxwell Demon framework.

## Directory Structure

Each validation run creates several files with timestamps:

```
validation_results_YYYYMMDD_HHMMSS.json  - Complete test results
summary_YYYYMMDD_HHMMSS.json             - Quick summary
front_sk_image_YYYYMMDD_HHMMSS.npy      - Front face S_k coordinates (8×8 grid)
back_sk_image_YYYYMMDD_HHMMSS.npy       - Back face S_k coordinates (8×8 grid)
front_info_image_YYYYMMDD_HHMMSS.npy    - Front face information density (8×8 grid)
back_info_image_YYYYMMDD_HHMMSS.npy     - Back face information density (8×8 grid)
test_pattern_YYYYMMDD_HHMMSS.npy        - Random test pattern (8×8)
carbon_copy_YYYYMMDD_HHMMSS.npy         - Conjugate carbon copy (8×8)
```

## File Contents

### validation_results_YYYYMMDD_HHMMSS.json

Complete validation results containing:
- Timestamp and metadata
- Results from all 6 tests:
  1. **Single Dual Pixel**: Creating and measuring a single dual-membrane pixel
  2. **Carbon Copy Propagation**: Changes propagating between front and back faces
  3. **Synchronized Evolution**: Both faces evolving together while maintaining conjugacy
  4. **Automatic Switching**: Time-based face switching at specified frequency
  5. **Dual Membrane Grid**: 8×8 grid of dual pixels with image measurements
  6. **Complementarity**: Demonstrating Heisenberg-like complementarity between faces

Each test result includes:
- Configuration parameters
- Measurements (categorical coordinates, densities, etc.)
- Verification checks
- Pass/fail status

### summary_YYYYMMDD_HHMMSS.json

Quick summary with:
- Total number of tests
- Number passed
- Overall pass/fail status
- Link to full results file

### Numpy Arrays (.npy files)

**S_k Images** (`front_sk_image`, `back_sk_image`):
- 8×8 arrays of S_k coordinates (categorical knowledge entropy)
- Front and back should be conjugate: `S_k_back ≈ -S_k_front` (for phase_conjugate)
- Load with: `np.load('front_sk_image_TIMESTAMP.npy')`

**Info Density Images** (`front_info_image`, `back_info_image`):
- 8×8 arrays of oscillatory information density
- Should be identical on both faces (physical property)
- Units: Hz × molecules/m³ (scaled by 10^-20)

**Pattern Arrays** (`test_pattern`, `carbon_copy`):
- Random test pattern and its conjugate transformation
- Demonstrates the "carbon copy" mechanism
- `carbon_copy ≈ -test_pattern` (for phase_conjugate)

## Loading and Analyzing Results

### Python Example

```python
import json
import numpy as np
from pathlib import Path

# Load results
results_dir = Path("observatory/src/maxwell/results/dual_membrane_validation")
timestamp = "20241126_123456"  # Replace with actual timestamp

# Load JSON results
with open(results_dir / f"validation_results_{timestamp}.json") as f:
    results = json.load(f)

print(f"All tests passed: {results['all_passed']}")

# Load images
front_sk = np.load(results_dir / f"front_sk_image_{timestamp}.npy")
back_sk = np.load(results_dir / f"back_sk_image_{timestamp}.npy")

# Verify conjugacy
print(f"S_k conjugacy check: {np.mean(front_sk + back_sk):.3f} ≈ 0")

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(front_sk, cmap='RdBu')
axes[0].set_title('Front Face S_k')
axes[1].imshow(back_sk, cmap='RdBu')
axes[1].set_title('Back Face S_k')
plt.show()
```

## Key Validation Points

### 1. Conjugate Relationship
- Front and back S_k coordinates sum to ~0 (for `phase_conjugate`)
- Maintained through evolution and switching

### 2. Carbon Copy Mechanism
- Changes on front propagate to back as conjugate changes
- `Δρ_front = +x` → `Δρ_back = -x`

### 3. Face Switching
- Observable face switches at specified frequency
- S_entropy state updates correctly
- Physical properties remain consistent

### 4. Complementarity
- Cannot access both faces simultaneously
- Hidden face has infinite uncertainty
- Respects categorical orthogonality

### 5. Grid Operation
- All pixels switch synchronously
- Categorical states differ, physical states identical
- Carbon copy transformation works across entire grid

## Troubleshooting

**If tests fail:**
1. Check the error message in the JSON results
2. Look for `validation_results_TIMESTAMP_FAILED.json` for partial results
3. Verify that S_k coordinates are initialized correctly
4. Check that switching updates both `dual_state` and `s_state`

**If images look wrong:**
1. Front and back S_k should have opposite signs (for phase_conjugate)
2. Info density should be identical on both faces
3. Carbon copy should be negative of test pattern (for phase_conjugate)

## Version History

- Initial implementation: 2024-11-26
- Added comprehensive result saving and intermediate data capture

---

For questions or issues, refer to the main dual-membrane documentation in `DUAL_MEMBRANE_README.md`.
