# Oscillatory Muscle Modeling Module

This module extends classical Hill-type muscle models with multi-scale oscillatory coupling theory to capture emergent coordination dynamics in muscle force generation and movement.

## Overview

Traditional Hill-type muscle models treat muscle force as a function of length, velocity, and activation. This module extends these models by incorporating **oscillatory coupling across 10 hierarchical scales**, from quantum membrane dynamics to allometric patterns.

## Key Features

### 1. Multi-Scale Oscillatory Hierarchy

The framework implements 10 hierarchical scales:

| Scale | Name | Frequency Range | Relevance to Muscle |
|-------|------|----------------|---------------------|
| 0 | Quantum Membrane | 10¹²-10¹⁵ Hz | Molecular dynamics |
| 1 | Intracellular | 10³-10⁶ Hz | Ca²⁺ signaling |
| 2 | Cellular | 10⁻¹-10² Hz | Action potentials |
| 3 | **Tissue** | 10⁻²-10¹ Hz | **Muscle fiber recruitment** |
| 4 | **Neural** | 1-100 Hz | **Motor unit firing** |
| 5 | **Neuromuscular** | 0.01-20 Hz | **Force generation** |
| 6 | **Cardiovascular** | 0.01-5 Hz | **Blood flow, fatigue** |
| 7 | **Locomotor** | 0.5-3 Hz | **Movement rhythm** |
| 8 | Circadian | 10⁻⁵ Hz | Daily patterns |
| 9 | Allometric | 10⁻⁸-10⁻⁵ Hz | Long-term adaptation |

**Bold scales** are most relevant for muscle dynamics and are emphasized in the model.

### 2. Coupling Strength Computation

The model computes coupling between scales using:

```
C_ij(t) = |1/T ∫₀ᵀ A_i(φ_j(t+τ)) e^(iφ_i(t+τ)) dτ|
```

Where:
- `A_i` is the amplitude of oscillation at scale `i`
- `φ_i`, `φ_j` are instantaneous phases
- `C_ij` quantifies how strongly scale `i` is entrained by scale `j`

### 3. Gear Ratio Transformations

Navigate between scales in O(1) complexity using:

```
R_{i→j} = ω_i / ω_j
```

This allows efficient analysis of how high-frequency dynamics (e.g., motor unit firing) translate to low-frequency outcomes (e.g., locomotor rhythm).

### 4. State Space Coordinates

The system state is characterized by three dimensions:

- **Knowledge (s_knowledge)**: Effective dimensionality of coupling structure
- **Time (s_time)**: Characteristic timescale of dynamics
- **Entropy (s_entropy)**: Complexity/predictability of behavior

### 5. Oscillatory Extensions to Hill Model

#### Classical Hill Model
```
F = F_max * f_l(L) * f_v(V) * a
```

#### Oscillatory-Enhanced Model
```
F = F_max * f_l(L) * f_v(V) * a * C_tissue * C_neuromuscular
```

Where `C_tissue` and `C_neuromuscular` are coupling-based modulation factors.

**Activation Dynamics with Neural Oscillations:**
```
da/dt = (u - a) / τ(a, u) * (1 + β * sin(ω_neural * t))
```

Neural oscillations (1-100 Hz) modulate activation/deactivation rates.

## Usage

### Basic Muscle Simulation

```python
from upward.muscle import OscillatoryMuscleModel

# Create muscle model
muscle = OscillatoryMuscleModel()

# Define excitation (step input)
def excitation(t):
    return 1.0 if 0.5 <= t <= 2.0 else 0.01

# Define muscle-tendon length (isometric contraction)
def muscle_tendon_length(t):
    return 0.31  # meters

# Simulate with oscillatory coupling
results = muscle.simulate_muscle_with_coupling(
    excitation_func=excitation,
    lmt_func=muscle_tendon_length,
    t_span=(0, 3.0),
    enable_coupling=True
)

# Extract results
time = results['time']
force = results['muscle_force']
activation = results['activation']
coupling_matrix = results['coupling_matrix']

# Compute performance metrics
metrics = muscle.compute_performance_metrics(results)
print(f"Peak force: {metrics['peak_force']:.2f} N")
print(f"Average coupling: {metrics['average_coupling']:.3f}")
```

### Body Segment Simulation

```python
from upward.muscle import LowerLimbModel

# Create lower limb model
model = LowerLimbModel(body_mass=70, height=1.75)

# Simulate gait cycle
results = model.simulate_gait_cycle(
    stride_frequency=1.5,  # Hz
    t_span=(0, 2.0)
)

# Extract joint angles and energies
angles = results['angles']  # Hip, knee, ankle
energies = results['energies']  # Oscillatory energy per segment
coupling = results['coupling_matrix']  # Segment coupling
```

### Coupling Analysis

```python
from upward.muscle import OscillatoryCouplingAnalyzer

analyzer = OscillatoryCouplingAnalyzer(sampling_rate=1000.0)

# Extract oscillatory components
signals = {}
for scale in muscle.scales:
    filtered = analyzer.bandpass_filter(
        force_signal, 
        scale.freq_min, 
        scale.freq_max
    )
    signals[scale.name] = filtered

# Compute coupling matrix
coupling_matrix = analyzer.compute_coupling_matrix(signals, muscle.scales)

# Analyze coupling strength
avg_coupling = np.mean(coupling_matrix)
print(f"Average inter-scale coupling: {avg_coupling:.3f}")
```

## Implementation Details

### `OscillatoryMuscleModel`

**Key Methods:**
- `simulate_muscle_with_coupling()`: Main simulation loop with oscillatory modulation
- `activation_dynamics_oscillatory()`: Neural oscillation-modulated activation
- `extract_oscillatory_components()`: Decompose force into multi-scale components
- `compute_performance_metrics()`: Calculate force, power, work, coupling efficiency

**State Variables:**
- Muscle fiber length `lm`
- Muscle fiber velocity `vm`
- Activation level `a`
- Coupling matrix `C_ij`
- State coordinates `(s_knowledge, s_time, s_entropy)`

### `OscillatoryKinematicChain`

Models body segments as **coupled pendular oscillators** with:
- Natural frequencies based on segment properties
- Damping based on tissue viscoelasticity
- Coupling based on proximity and frequency resonance

**Equations of Motion:**
```
θ̈_i = (1/I_i) * [τ_external_i + τ_coupling_i - c_i * θ̇_i]

τ_coupling_i = Σ_j C_ij * k_j * (θ_j - θ_i)
```

### `LowerLimbModel`

Combines thigh, shank, and foot segments with simplified muscle torques to simulate gait dynamics with oscillatory coupling.

## Comparison with Classical Models

| Aspect | Classical Hill | Oscillatory Hill |
|--------|----------------|------------------|
| Force generation | Static parameters | Dynamic coupling modulation |
| Activation | 1st-order ODE | Neural oscillation-modulated |
| Coordination | Not addressed | Multi-scale coupling |
| Fatigue | Phenomenological | Emergent from coupling decay |
| Analysis complexity | O(n) per scale | O(1) with gear ratios |

## Theoretical Foundation

This implementation is based on the multi-scale oscillatory coupling framework described in the main project README. Key theoretical concepts:

1. **Emergence from Coupling**: Complex muscle behavior emerges from coupling across hierarchical scales rather than being explicitly programmed.

2. **Decoupling = Dysfunction**: Reduced coupling strength indicates pathology or fatigue.

3. **Gear Ratio Navigation**: High-frequency processes (molecular) can be understood through their coupling ratios to low-frequency outcomes (movement).

4. **State Space Characterization**: The tri-dimensional state space captures system knowledge, temporal scale, and entropy simultaneously.

## Examples

Run the example simulations:

```bash
# Muscle simulation with coupling analysis
cd upward/muscle
python muscle_model.py

# Body segment simulation  
python body_segmentation.py
```

This generates plots comparing:
- Force with vs. without coupling
- Activation dynamics with neural modulation
- Coupling matrices across scales
- State space trajectories
- Joint angles and energies during gait

## Extensions and Future Work

Potential extensions to this framework:

1. **Muscle Fatigue**: Model as progressive decoupling
2. **Injury Recovery**: Track re-coupling dynamics
3. **Training Adaptation**: Long-term coupling strengthening
4. **Multi-Muscle Coordination**: Inter-muscle coupling matrices
5. **Surface Compliance**: Ground reaction force coupling
6. **Sensorimotor Integration**: Coupling to proprioceptive feedback

## References

**Classical Muscle Models:**
- Thelen DG (2003). Adjustment of muscle mechanics model parameters. J Biomech Eng.
- McLean SG et al. (2003). 3-D model to predict knee joint loading. J Biomech Eng.
- Zajac FE (1989). Muscle and tendon properties, models, scaling. Crit Rev Biomed Eng.

**Body Segment Parameters:**
- de Leva P (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. J Biomech.

**Oscillatory Theory:**
- See main project README and `docs/oscillations/` for complete theoretical framework.

## License

See project LICENSE file.

## Author

Kundai Farai Sachikonye  
Based on oscillatory coupling theory framework

