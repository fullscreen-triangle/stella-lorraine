# Quick Start - Fix and Test Experiments

## 🚀 What to Do Now

Follow these steps in order:

---

## Step 1: Run Diagnostics (Find What's Broken)

```bash
cd observatory/src
python diagnose_experiments.py
```

**This will show you:**
- ✓ Which experiments are ready
- ⚠️  Which need a `main()` function
- ✗ Which have import errors

---

## Step 2: Test One Working Experiment

```bash
# Test the recursive observer (should work since we just updated it)
python navigation/gas_molecule_lattice.py
```

**Expected:**
- Script runs
- Creates `results/recursive_observers/` directory
- Saves JSON and PNG files
- Shows progress [1/5] through [5/5]

---

## Step 3: Fix Failed Experiments One by One

For each failed experiment from diagnostics:

### Common Fix #1: Remove Bad Imports

If you see:
```
cannot import name 'ThermodynamicState' from 'simulation.Wave'
```

**Fix:** Open the file and remove/comment out the bad import:
```python
# Remove this:
# from simulation.Wave import ThermodynamicState

# Or comment it out:
## from simulation.Wave import ThermodynamicState
```

### Common Fix #2: Add Missing main() Function

If experiment has "NO main()", add this at the end:

```python
def main():
    """Main experimental function"""
    # Your existing code here
    # Make sure to return results
    return results_dict, figure_path

if __name__ == "__main__":
    main()
```

---

## Step 4: Test Each Fixed Experiment

After fixing an experiment, test it:

```bash
python navigation/experiment_name.py
```

**Verify:**
- [ ] Runs without errors
- [ ] Creates JSON in `results/`
- [ ] Creates PNG figure
- [ ] Shows final "✨ Experiment complete!" message

---

## Step 5: Run Validation Suite (Once Most Work)

When at least 7-8 experiments work individually:

```bash
python run_validation_suite.py
```

This will:
- Run each experiment as separate process
- Collect all results
- Generate summary report
- Create combined figure

---

## Troubleshooting

### Q: "Module not found"
**A:** Make sure you're in `observatory/src` directory

### Q: "No such file or directory"
**A:** Check the file actually exists:
```bash
ls navigation/gas_molecule_lattice.py
```

### Q: Import errors
**A:** Remove or fix the problematic imports in the file

### Q: No output files
**A:** Check `results/` directory exists:
```bash
mkdir -p ../results
```

---

## Quick Command Reference

```bash
# 1. Diagnose all experiments
python diagnose_experiments.py

# 2. Test single experiment
python navigation/gas_molecule_lattice.py

# 3. Test another experiment
python navigation/harmonic_network_graph.py

# 4. Run full validation suite
python run_validation_suite.py

# 5. Check results
ls -R ../results/
```

---

## Expected Final Structure

```
results/
├── recursive_observers/
│   ├── recursive_observers_20251010_071234.json
│   └── recursive_observers_20251010_071234.png
├── harmonic_network/
│   ├── harmonic_network_20251010_071456.json
│   └── harmonic_network_20251010_071456.png
├── validation_reports/
│   ├── validation_report_20251010_080000.json
│   └── validation_summary_20251010_080000.png
└── ... (other experiments)
```

---

## Priority Order (Fix These First)

1. **gas_molecule_lattice.py** ← Already updated ✓
2. **harmonic_network_graph.py** ← Already updated ✓
3. harmonic_extraction.py
4. fourier_transform_coordinates.py
5. entropy_navigation.py
6. (rest can wait)

---

## Success Criteria

✅ Each experiment runs independently
✅ Each creates JSON + PNG output
✅ Validation suite gets >80% success rate
✅ Ready for publication

---

## Next Steps

1. Run `diagnose_experiments.py` now
2. Note which experiments failed
3. Fix them one by one
4. Test each after fixing
5. Run validation suite when ready

Good luck! 🚀
