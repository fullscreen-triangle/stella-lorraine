# Validation Scripts Guide

## All Validation Scripts (In Order)

Run these scripts to validate the complete Pixel Maxwell Demon framework:

###1. **Complete System Validation** (Original Pixel Demon)
```bash
cd observatory/src/maxwell
python validate_complete_system.py
```

**What it tests:**
- Pixel Maxwell Demon basics
- Virtual detector cross-validation
- Harmonic coincidence networks
- Reflectance cascade
- Categorical rendering (3D)
- Trans-Planckian precision
- Live cell imaging

**Results saved to:**
- `results/maxwell_validation/validation_TIMESTAMP.json`

---

### 2. **Dual-Membrane Validation** (Front/Back Conjugate States)
```bash
cd observatory/src/maxwell
python validate_dual_membrane.py
```

**What it tests:**
- Single dual-membrane pixel
- Carbon copy propagation
- Synchronized dual evolution
- Automatic face switching
- Dual-membrane grid (8×8)
- Complementarity (Heisenberg-like)

**Results saved to:**
- `results/dual_membrane_validation/validation_results_TIMESTAMP.json`
- `results/dual_membrane_validation/summary_TIMESTAMP.json`
- `results/dual_membrane_validation/front_sk_image_TIMESTAMP.npy`
- `results/dual_membrane_validation/back_sk_image_TIMESTAMP.npy`
- `results/dual_membrane_validation/front_info_image_TIMESTAMP.npy`
- `results/dual_membrane_validation/back_info_image_TIMESTAMP.npy`
- `results/dual_membrane_validation/test_pattern_TIMESTAMP.npy`
- `results/dual_membrane_validation/carbon_copy_TIMESTAMP.npy`

---

### 3. **Circuit Representation Validation** (Electrical Circuit Analogy)
```bash
cd observatory/src/maxwell
python validate_circuit_representation.py
```

**What it tests:**
- Simple circuit from S-coordinates
- Pixel demon → electrical circuit conversion
- Ammeter/voltmeter complementarity
- Power conservation
- Circuit diagrams
- Impedance calculation
- Kirchhoff's laws
- SPICE export

**Results saved to:**
- `results/circuit_validation/validation_results_TIMESTAMP.json`
- `results/circuit_validation/summary_TIMESTAMP.json`

---

### 4. **Image Processing Validation** (3D/JPEG Images) ⭐ NEW
```bash
cd observatory/src/maxwell

# Without real image (synthetic only)
python validate_image_processing.py

# With real JPEG/PNG image
python validate_image_processing.py --image path/to/your/image.jpg
```

**What it tests:**
- Synthetic image processing (64×64 RGB gradient)
- Real JPEG/PNG image processing (if provided)
- Carbon copy transformation on images
- Front/back face extraction
- S_k coordinate mapping

**Results saved to:**
- `results/image_processing/validation_results_TIMESTAMP.json`
- `results/image_processing/synthetic_input_TIMESTAMP.png`
- `results/image_processing/synthetic_front_TIMESTAMP.npy`
- `results/image_processing/synthetic_back_TIMESTAMP.npy`
- `results/image_processing/synthetic_front_sk_TIMESTAMP.npy`
- `results/image_processing/synthetic_back_sk_TIMESTAMP.npy`
- `results/image_processing/real_front_TIMESTAMP.npy` (if real image provided)
- `results/image_processing/real_back_TIMESTAMP.npy` (if real image provided)
- `results/image_processing/pattern_TIMESTAMP.npy`
- `results/image_processing/carbon_copy_TIMESTAMP.npy`

---

## Quick Start: Run All Validations

```bash
cd observatory/src/maxwell

# 1. Original system
python validate_complete_system.py

# 2. Dual membrane
python validate_dual_membrane.py

# 3. Circuit representation
python validate_circuit_representation.py

# 4. Image processing (with your image)
python validate_image_processing.py --image /path/to/image.jpg
```

---

## What Gets Saved

### All Scripts Save:
✅ **JSON results** with complete test data
✅ **Timestamps** for tracking validation runs
✅ **Summary files** for quick review
✅ **All intermediate data** (numpy arrays, images, etc.)

### Result Locations:

```
observatory/src/maxwell/results/
├── maxwell_validation/          # Script 1 results
│   └── validation_TIMESTAMP.json
├── dual_membrane_validation/    # Script 2 results
│   ├── validation_results_TIMESTAMP.json
│   ├── summary_TIMESTAMP.json
│   ├── front_sk_image_TIMESTAMP.npy
│   ├── back_sk_image_TIMESTAMP.npy
│   └── ...
├── circuit_validation/          # Script 3 results
│   ├── validation_results_TIMESTAMP.json
│   └── summary_TIMESTAMP.json
└── image_processing/            # Script 4 results (⭐ FOR 3D/JPEG)
    ├── validation_results_TIMESTAMP.json
    ├── synthetic_input_TIMESTAMP.png
    ├── synthetic_front_TIMESTAMP.npy
    ├── synthetic_back_TIMESTAMP.npy
    └── ...
```

---

## For 3D/JPEG Image Testing

**Script 4 (`validate_image_processing.py`) is specifically for testing with images!**

### Usage:

```bash
# Test with synthetic image only (no dependencies needed)
python validate_image_processing.py

# Test with your JPEG/PNG image
python validate_image_processing.py --image ~/Pictures/test.jpg

# Test with any image format supported by PIL
python validate_image_processing.py --image /path/to/image.png
```

### What It Does:

1. **Synthetic Test**: Creates a 64×64 RGB gradient image and processes it through the dual-membrane grid
2. **Real Image Test**: Loads your JPEG/PNG, converts to grayscale, downsamples if needed, and processes through dual-membrane
3. **Carbon Copy Test**: Demonstrates the conjugate transformation on a random pattern

### Output:

All results are saved as:
- **`.npy` files** - Numpy arrays you can load and analyze
- **`.png` files** - Images you can view directly
- **`.json` files** - Metadata and validation results

### Loading Results:

```python
import numpy as np
import json
from pathlib import Path

results_dir = Path("observatory/src/maxwell/results/image_processing")

# Load validation results
with open(results_dir / "validation_results_TIMESTAMP.json") as f:
    results = json.load(f)

# Load front face image
front_image = np.load(results_dir / "synthetic_front_TIMESTAMP.npy")

# Load S_k coordinates
front_sk = np.load(results_dir / "synthetic_front_sk_TIMESTAMP.npy")
back_sk = np.load(results_dir / "synthetic_back_sk_TIMESTAMP.npy")

# Verify conjugate relationship
print(f"Conjugate check: {np.mean(front_sk + back_sk):.3f} ≈ 0")
```

---

## Dependencies

### Required (all scripts):
- `numpy`
- `logging`
- `json`
- `pathlib`

### Optional (for image processing):
- `PIL` (Pillow) - for JPEG/PNG loading
- `opencv-python` - alternative image loader
- `scikit-image` - for downsampling large images

Install optional dependencies:
```bash
pip install pillow opencv-python scikit-image
```

---

## Troubleshooting

### "No module named 'PIL'"
Install Pillow:
```bash
pip install pillow
```

### "Image too large, taking forever"
The script automatically downsamples images larger than 128×128. For even faster processing:
```bash
# Resize your image first using ImageMagick or similar
convert large_image.jpg -resize 64x64 small_image.jpg
python validate_image_processing.py --image small_image.jpg
```

### "Script runs but no results saved"
Check that you have write permissions to the `results/` directory. The script creates directories automatically but may fail if permissions are restrictive.

### "Test failed with serialization error"
The scripts now use `NumpyEncoder` to handle numpy types. If you still see serialization errors, check that you're using the updated scripts.

---

## Summary Table

| Script | Purpose | Results Location | 3D/Image Test? |
|--------|---------|------------------|----------------|
| `validate_complete_system.py` | Original pixel demon framework | `results/maxwell_validation/` | Partial (3D rendering) |
| `validate_dual_membrane.py` | Dual-membrane front/back states | `results/dual_membrane_validation/` | Yes (8×8 grids) |
| `validate_circuit_representation.py` | Electrical circuit analogy | `results/circuit_validation/` | No |
| **`validate_image_processing.py`** | **3D/JPEG image processing** | **`results/image_processing/`** | **✅ YES - Full image support** |

---

## Next Steps

After running validations:

1. **Review JSON results** to see all test data
2. **Load numpy arrays** to analyze front/back face relationships
3. **Verify conjugate relationships** (front + back ≈ 0 for S_k)
4. **Visualize images** using matplotlib or similar
5. **Compare results** across different transform types (phase_conjugate, harmonic, etc.)

Example visualization:
```python
import matplotlib.pyplot as plt
import numpy as np

front = np.load("results/image_processing/synthetic_front_sk_TIMESTAMP.npy")
back = np.load("results/image_processing/synthetic_back_sk_TIMESTAMP.npy")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(front, cmap='RdBu')
axes[0].set_title('Front Face S_k')
axes[1].imshow(back, cmap='RdBu')
axes[1].set_title('Back Face S_k')
axes[2].imshow(front + back, cmap='gray')
axes[2].set_title('Sum (should ≈ 0)')
plt.show()
```

---

**For questions or issues, see the main documentation in `DUAL_MEMBRANE_README.md` and `CIRCUIT_COMPLEMENTARITY.md`.**
