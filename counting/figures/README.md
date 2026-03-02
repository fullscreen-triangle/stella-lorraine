# Validation Figure Panels

This directory contains publication-quality panel figures generated from the categorical state counting validation experiments.

## Overview

**Total Figures**: 24 panel charts (6 panels × 4 validation runs)
**Format**: PNG at 300 DPI
**Layout**: 4 charts per panel (12" × 3")
**3D Charts**: At least one 3D visualization per panel

## Panel Descriptions

### Panel 1: Hardware Oscillator Validation

**File**: `*_panel_1_oscillator.png`

Validates the fundamental identity: **dM/dt = ω/(2π/M) = 1/⟨τ_p⟩**

- **Chart A**: Cycle Count vs Duration (log-log) - Shows linear relationship between time and oscillator cycles
- **Chart B**: Reconstruction Error - Demonstrates error < 10⁻¹⁰ across all measurements
- **Chart C**: **3D Surface** - Time × Frequency × Cycles showing the fundamental counting relationship
- **Chart D**: Fundamental Identity Verification - Perfect agreement between expected and measured cycles

**Key Result**: Time measurement and state counting are equivalent operations (100% agreement).

---

### Panel 2: Partition Coordinates Validation

**File**: `*_panel_2_partition.png`

Validates CatScript claim: **Partition coordinates (n, ℓ, m, s) from oscillator counts**

- **Chart A**: Principal Quantum Number n vs State Count M - Shows √(M/2) relationship
- **Chart B**: Capacity Formula C(n) = 2n² - Confirms quadratic scaling
- **Chart C**: **3D Scatter** - (n, ℓ, m) coordinate space colored by capacity
- **Chart D**: Cumulative Capacity - Validates N(n) = n(n+1)(2n+1)/3

**Key Result**: Capacity formula validated with 100% agreement across all state counts.

---

### Panel 3: Categorical Temperature Validation

**File**: `*_panel_3_temperature.png`

Validates categorical cryogenics: **T = 2E/(3k_B × M)**

- **Chart A**: T_categorical vs M (log-log) - Shows 1/M temperature suppression
- **Chart B**: Suppression Factor - Confirms exact 1/M scaling
- **Chart C**: **3D Surface** - T(E, M) showing temperature landscape across energy and state count
- **Chart D**: Measured vs Expected Temperature - Perfect diagonal agreement

**Key Result**: Categorical temperature shows 1/M suppression with 100% formula agreement.

---

### Panel 4: Ion Trajectory Validation

**File**: `*_panel_4_trajectory.png`

Validates complete ion journey as state counting sequence

- **Chart A**: Total State Count vs m/z - Shows state accumulation through MS1 journey
- **Chart B**: Journey Time vs m/z - Temporal progression through instrument
- **Chart C**: **3D Scatter** - (m/z, charge, state_count) colored by kinetic energy
- **Chart D**: Stage Breakdown - Horizontal bar chart showing state counts per journey stage

**Key Result**: All ion trajectories complete with validated state counting at each stage.

---

### Panel 5: Pipeline Validation

**File**: `*_panel_5_pipeline.png`

Validates pipeline processing of real mass spectrometry data

- **Chart A**: State Count Distribution - Histogram showing tight clustering around mean
- **Chart B**: Capacity Bounds Check - All ions fall below capacity limit (M/C < 1)
- **Chart C**: **3D Scatter** - (m/z, n, state_count) colored by total capacity
- **Chart D**: Selection Rules Validation - Bar chart showing n values colored by validity

**Key Result**: 100% of ions occupy states within bounded phase space capacity.

---

### Panel 6: Statistical Analysis

**File**: `*_panel_6_statistics.png`

Statistical summary across all validation metrics

- **Chart A**: Trans-Planckian Statistics - Mean, std, min, max state counts
- **Chart B**: CatScript Statistics - Principal quantum number distribution
- **Chart C**: **3D Scatter** - Temperature distribution cloud showing T vs suppression vs sample index
- **Chart D**: Regime Distribution (pie chart) or Validation Fractions (bar chart)

**Key Result**: Statistical consistency across 6,855 ions with minimal variance.

---

## Data Sources

Four validation result files were processed:

1. **20260301_041104** - Early synthetic run
2. **20260301_041127** - Refined synthetic run
3. **20260301_041230** - Intermediate run
4. **20260301_041532** - **Final run with real mzML data (6,855 ions)** ← Use this for publication

## Recommended Figures for Paper

For the manuscript, use panels from **validation_results_20260301_041532** as these contain:
- Real experimental data (not synthetic)
- Largest ion count (6,855 ions)
- Most robust statistics

### Suggested Figure Numbers

- **Figure 1**: Panel 1 (Oscillator) - Establishes fundamental counting identity
- **Figure 2**: Panel 2 (Partition) - Validates CatScript coordinate framework
- **Figure 3**: Panel 3 (Temperature) - Demonstrates categorical cryogenics
- **Figure 4**: Panel 5 (Pipeline) - Shows real data validation
- **Figure S1** (Supplementary): Panel 4 (Trajectory)
- **Figure S2** (Supplementary): Panel 6 (Statistics)

## Visual Design Features

### Color Scheme
- **Primary blue**: #2E86AB (Trans-Planckian)
- **Magenta**: #A23B72 (CatScript)
- **Orange**: #F18F01 (Categorical Cryogenics)
- **Red**: #C73E1D (Invalid/thresholds)
- **Colormap**: Viridis (for continuous data)

### Typography
- Figure labels: **A, B, C, D** (bold, left-aligned)
- Font size: 8pt base, 9pt labels, 10pt titles
- Minimal text - data speaks for itself

### 3D Visualizations
- Standard viewing angle: elevation=20°, azimuth=45°
- White edge colors for depth perception
- Alpha transparency for overlapping points

## File Naming Convention

```
validation_results_YYYYMMDD_HHMMSS_panel_N_description.png
```

Where:
- `YYYYMMDD_HHMMSS`: Timestamp of validation run
- `N`: Panel number (1-6)
- `description`: Short name (oscillator, partition, temperature, etc.)

## Regenerating Figures

To regenerate all panels:

```bash
cd counting
python generate_validation_figures.py --all --output-dir figures
```

To regenerate from specific validation result:

```bash
python generate_validation_figures.py validation_results/validation_results_20260301_041532.json --output-dir figures
```

## Technical Specifications

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG with transparency support
- **Dimensions**: 12" × 3" (suitable for two-column format)
- **Color space**: RGB
- **Grid**: Light gray (#808080, alpha=0.3) with dashed lines
- **Markers**: White-filled with colored edges (width=2pt)
- **Line width**: 2pt for primary data, 1pt for theory/references

## Integration with LaTeX

To include in your LaTeX manuscript:

```latex
\begin{figure*}[!htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/validation_results_20260301_041532_panel_1_oscillator.png}
    \caption{\textbf{Hardware Oscillator Validation: Fundamental counting identity.}
    (\textbf{A}) Cycle count scales linearly with duration across 6 orders of magnitude.
    (\textbf{B}) Reconstruction error below $10^{-10}$ confirms exact time-counting equivalence.
    (\textbf{C}) 3D surface shows fundamental relationship: Cycles = Time × Frequency.
    (\textbf{D}) Perfect agreement between expected and measured cycles validates dM/dt = 1/⟨τ_p⟩.}
    \label{fig:oscillator_validation}
\end{figure*}
```

## Quality Assurance

All figures have been validated for:
- ✓ Data accuracy (matches JSON sources)
- ✓ Scientific correctness (formulas, units, scales)
- ✓ Visual clarity (readable at publication size)
- ✓ Consistency (color scheme, typography, layout)
- ✓ Accessibility (color-blind friendly viridis colormap)

## Contact

For questions about figure generation:
- **Script**: `generate_validation_figures.py`
- **Author**: Kundai Farai Sachikonye
- **Institution**: Technical University of Munich

---

**Generated**: 2026-03-01
**Total Size**: 7.2 MB (24 figures)
**Status**: Ready for publication
