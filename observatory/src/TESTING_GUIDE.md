# Testing Guide - Step by Step

## 🔍 How to Test Each Experiment Independently

Follow these steps to verify everything works correctly before running the full validation suite.

---

## Step 1: Test Single Experiment

### Option A: Direct Script Execution

```bash
cd observatory/src
python navigation/gas_molecule_lattice.py
```

**Expected output:**
```
======================================================================
   EXPERIMENT: RECURSIVE OBSERVER NESTING
   Trans-Planckian Precision Through Fractal Observation
======================================================================

   Timestamp: 20251010_HHMMSS
   Results directory: ../results/recursive_observers

[1/5] Initializing molecular lattice...
...
[5/5] Saving results and generating visualizations...
   ✓ Results saved: ../results/recursive_observers/recursive_observers_TIMESTAMP.json
   ✓ Figure saved: ../results/recursive_observers/recursive_observers_TIMESTAMP.png

✨ Experiment complete!
```

### Option B: Test Script

```bash
cd observatory/src
python test_single_experiment.py
```

---

## Step 2: Check Output Files

After running, verify these files exist:

```bash
# Check results directory
ls -la ../results/recursive_observers/

# Should see:
# recursive_observers_TIMESTAMP.json
# recursive_observers_TIMESTAMP.png
```

---

## Step 3: Validate JSON Format

```bash
# Open JSON file in text editor or:
python -m json.tool ../results/recursive_observers/recursive_observers_TIMESTAMP.json
```

**Expected structure:**
```json
{
  "timestamp": "...",
  "experiment": "recursive_observer_nesting",
  "configuration": {...},
  "recursion_results": {...},
  "transcendent_results": {...},
  "planck_analysis": {...}
}
```

---

## Step 4: View Figure

Open the PNG file - should see a 6-panel layout:
1. Precision cascade plot
2. Active observers bar chart
3. Observation paths line plot
4. Planck comparison horizontal bar chart
5. FFT spectrum
6. Summary text box

---

## Step 5: Test Other Experiments One by One

Test each script individually to identify which ones work:

```bash
# Test harmonic network graph
python navigation/harmonic_network_graph.py

# Test harmonic extraction
python navigation/harmonic_extraction.py

# Test molecular vibrations
python navigation/molecular_vibrations.py

# Test SEFT
python navigation/fourier_transform_coordinates.py

# Test entropy navigation
python navigation/entropy_navigation.py

# Test miraculous measurement
python navigation/multidomain_seft.py

# Test finite observer
python navigation/finite_observer_verification.py

# Test molecular clock
python simulation/Molecule.py

# Test gas chamber
python simulation/GasChamber.py
```

---

## Step 6: Check for Missing Dependencies

If any script fails, check for missing imports:

```python
# Add at top of script to debug:
import sys
print("Python path:", sys.path)
print("Current dir:", os.getcwd())

# Check imports work:
try:
    import numpy
    print("✓ numpy")
except:
    print("✗ numpy - install with: pip install numpy")

try:
    import matplotlib
    print("✓ matplotlib")
except:
    print("✗ matplotlib - install with: pip install matplotlib")
```

---

## Step 7: Fix Import Errors

If you see import errors like:
```
cannot import name 'ThermodynamicState' from 'simulation.Wave'
```

This means the script is trying to import something that doesn't exist. Options:

### Fix A: Remove Bad Imports
Edit the script and remove/comment out the problematic import

### Fix B: Create Stub Functions
Add missing functions as simple stubs:

```python
# If script needs ThermodynamicState but it doesn't exist:
class ThermodynamicState:
    """Stub class for compatibility"""
    pass
```

---

## Step 8: Run Validation Suite (Once Individual Scripts Work)

Only run the full suite after confirming individual scripts work:

```bash
python run_validation_suite.py
```

---

## Common Issues & Fixes

### Issue 1: "Module not found"

**Cause:** Python can't find the module

**Fix:**
```bash
# Make sure you're in the src directory
cd observatory/src

# Check Python can see modules:
python -c "import sys; print('\n'.join(sys.path))"
```

### Issue 2: "cannot import name 'X'"

**Cause:** Function/class doesn't exist in module

**Fix:**
1. Open the module file
2. Check what's actually defined with `__all__` or look for `def` and `class`
3. Remove imports of non-existent items
4. OR create stub definitions

### Issue 3: "No module named 'matplotlib'"

**Cause:** Missing dependency

**Fix:**
```bash
pip install matplotlib numpy scipy
```

### Issue 4: Scripts run but no output files

**Cause:** Results directory not created or wrong path

**Fix:**
```python
# Check/create results directory
import os
results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory: {os.path.abspath(results_dir)}")
```

---

## Debugging Template

Add this to any failing script to debug:

```python
#!/usr/bin/env python3
import sys
import os

print("="*70)
print("DEBUG INFO")
print("="*70)
print(f"Python: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")
print(f"Python path: {sys.path[:3]}")
print("="*70)

# Rest of your script...
```

---

## Success Checklist

Before considering a script "working":

- [ ] Script runs without errors
- [ ] JSON file created in `results/` subdirectory
- [ ] PNG figure created (if applicable)
- [ ] JSON is valid (can be loaded)
- [ ] Figure displays 6 panels
- [ ] Timestamp in filenames
- [ ] Results directory auto-created

---

## Next Steps

Once individual scripts work:

1. ✅ Test each experiment standalone
2. ✅ Verify output files created
3. ✅ Check JSON format
4. ✅ View figures
5. ✅ Run validation suite
6. ✅ Review validation report
7. ✅ Ready for publication!

---

## Getting Help

If scripts still fail after these steps:

1. Check which specific import is failing
2. Look at the actual error message (not just the first line)
3. Verify the file/function actually exists
4. Check file paths are correct
5. Ensure you're in the right directory

The key principle: **Each script should run completely independently with just `python script_name.py`**
