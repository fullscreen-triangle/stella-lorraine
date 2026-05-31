# Monograph Corpus: Complete Arrangement and Source Index

**Title**: On the Categorical Mechanics of Bounded Phase Space:
From a Single Axiom to a Running LHC Trigger

**Structure**: 20 papers across 5 volumes
**Principle**: Every paper derives from the single Bounded Phase Space Axiom.
  Every volume is independently readable by its target audience.
  The entire structure is one deductive chain stated at every scale of nature.

**Status key**:
- PRESENT = file already exists in this repository
- FETCH   = file exists in another repository, needs to be copied here
- WORKING = currently being written

---

## VOLUME I — Mathematical Physics Foundation
*Target audience: mathematical physicists, foundations of physics*
*Core claim: one axiom → all of physics*

### Paper 1: The Bounded Phase Space Law
**Role**: Master deductive framework. Single axiom → 30+ consequences with complete proofs.
  Derives: oscillatory necessity, partition coordinates (n,l,m,s), C(n)=2n²,
  selection rules, Pauli exclusion, Lorentz invariance, c, E=hbar*omega, E=mc²,
  charge emergence, ideal gas, transport coefficients, nuclear magic numbers,
  atomic structure, dimensional reduction, composition-inflation T(n,d),
  three routes to G, framework closure, MOND, dark energy w=-0.75, secular dG/Gdt.
**Status**: PRESENT
**Source**: `trans_planckian/publications/sources/bounded-phase-space-trajectory.tex`
**Size**: 476 KB (~5000 lines) — the largest paper in the corpus
**Key theorems**: Poincaré recurrence, oscillatory necessity, C(n)=2n², T(n,d)=d(d+1)^{n-1},
  three routes to G with (d+1)^{-n} convergence, framework closure

### Paper 2: Composition-Inflation and Three Routes to G
**Role**: Focused extraction of the counting mechanism and G derivation.
  Derives T(n,d)=d(d+1)^{n-1} from integer compositions + dimensional labeling.
  Proves Planck depth n_P formula. Derives all SI constants in categorical angular units.
  Shows G is the unique irreducible constant.
**Status**: PRESENT
**Source**: `chitungwiza/physics/composition-inflation.tex`
**Key theorems**: T(n,d)=d(d+1)^{n-1}, Planck depth n_P=56 for caesium,
  c=2pi rad/tick, hbar=E_tick/(2pi), k_B=E_tick/ln(d+1), G irreducible
**Companion**: `trans_planckian/publications/sources/universal-shader-depth.tex`
  (USD — standalone G-derivation paper with full numerical validation,
  three routes converging to (d+1)^{-n} precision, CODATA agreement at n=8)
**Status companion**: PRESENT

---

## VOLUME II — Partition Spectroscopy at Molecular Scale
*Target audience: mass spectrometrists, physical chemists, biophysicists*
*Core claim: a mass spectrometer is a partition detector; mass is partition memory*

### Paper 3: Partition Lagrangian and the Ion as Partition Malformation
**Role**: Derives the partition Lagrangian L_M = ½μ|ẋ|² + μẋ·A_M − M(x,t).
  Shows all four analyzer types (TOF, quadrupole, Orbitrap, FT-ICR) follow
  from this single Lagrangian. Derives mass from three independent routes.
  Validated: 4545 NIST entries + 127000 proteomic bijective transformations.
**Status**: PRESENT
**Source**: `trans_planckian/publications/sources/ion-trajectory-completion-mechanism.tex`
**Key results**: Lorentz force derived not assumed; mass = accumulated partition residue;
  E=mc² as theorem; equivalence principle as geometric identity;
  charge = partition boundary attribute

### Paper 4: Five Thermodynamic Regimes from the Ion Journey
**Role**: Shows the ion journey through a mass spectrometer traverses ALL five
  canonical equations of state. Universal EOS: PV = NkT·S(V,N,T,{n_i,l_i,m_i,s_i}).
  C(n)=2n² appears explicitly in the degenerate matter structural factor.
  Validated: 5277 ions classified across all five regimes.
**Status**: PRESENT
**Source**: `trans_planckian/publications/mass-spec/ion-thermodynamic-regimes.tex`
**Key results**: Ideal/Plasma/Degenerate/Relativistic/BEC from one framework;
  universal EOS; five regimes correspond to five LHC detector regions

### Paper 5: Categorical State Counting and the Fundamental Identity
**Role**: dM/dt = ω/(2π) = 1/⟨τ_p⟩ — time = partition counting.
  Proves heat-entropy decoupling: Cov(δQ, dS_cat) = 0.
  Counting irreversibility: P_reverse ~ exp(-N_state).
  QND: partition coordinates are conserved observables.
  Bijective validation via physics constraints (Weber, Reynolds, Ohnesorge).
**Status**: PRESENT
**Source**: `trans_planckian/publications/mass-spec/categorical-state-counting.tex`
**Key results**: Fundamental identity; heat-entropy decoupled (signal and noise orthogonal);
  counting is inherently irreversible; categorical aperture at zero thermodynamic cost

### Paper 6: Mass Computing — Ternary Addresses and Partition Determinism
**Role**: Ternary address (k trits) = position AND trajectory simultaneously.
  Partition Determinism: address → spectrum without dynamic simulation.
  MassScript domain-specific language for virtual experiments.
  Validated: 96.3% accuracy on 4271 compounds, 10^6× speedup over physical measurement.
**Status**: PRESENT
**Source**: `trans_planckian/publications/mass-spec/mass-partitioning-computing.tex`
**Key results**: T(n,d) as trit-cell count; trajectory-position equivalence;
  address IS trajectory; partition synthesis replaces simulation;
  10^8× speedup in Rust implementation

### Paper 7: Single-Ion Thermodynamics and QND Multi-Detector Measurement
**Role**: Resolves single-particle thermodynamics paradox: PV = k_B·T_cat for N=1.
  Categorical temperature T_cat = hbar·ω/(2π·k_B).
  Complete commutation: [n̂,l̂] = [l̂,m̂] = [m̂,ŝ] = 0.
  All four partition coordinates measured simultaneously without backaction.
  Validated: 847 compounds, backaction suppression to Δp/p ~ 10^{-3}.
**Status**: PRESENT
**Source**: `trans_planckian/publications/mass-spec/quantupartite-ion-observatory.tex`
**Key results**: Single-ion ideal gas PV=k_BT_cat; QND multi-detector;
  all four detector subsystems measure commuting observables simultaneously;
  this is why LHC L1 trigger can combine all subsystems in one latency window

### Paper 8: Light, Fluids, and Chromatography as Partition Geometry
**Role**: Three phenomena (light, viscosity, retention) unified by partition lag τ_c.
  c = Δx/τ_c (speed of light from partition propagation).
  μ = τ_c × g (viscosity from lag × coupling).
  τ_p = ℏ/ΔE (universal transport formula).
  Validated: viscosity 2% error, retention 3.2% error, UV-Vis 1-6% error.
**Status**: PRESENT
**Source**: `trans_planckian/publications/sources/mass-transfer-mechanism.tex`
**Key results**: c=Δx/τ_c gives LHC bunch crossing window 25ns = c/7.5m;
  timing window is partition propagation time, not engineering choice;
  universal transport unifies all detector response times

---

## VOLUME IIb — Categorical Optics: One Beam, Complete Scene
*Target audience: spectroscopists, optical physicists, instrument designers*
*Core claim: "bending light without mirrors" — one beam explains a whole scene*

### Paper 9: Harmonic Scattering Loops and Multi-Source Reconstruction
**Role**: One-ray-per-source constraint is not fundamental.
  Transfer matrix rank = C+1 where C = cycle rank of molecular harmonic graph.
  Capacity: N_max = (C+1)·T_deph/T_L sources from single optical path.
  Molecular resonator creates virtual paths via harmonic loops — "bends light"
  without any physical bending or mirrors.
  Validated: benzene-like resonator (C=3) reconstructs 4 sources, e=2.8×10^{-15}.
**Status**: PRESENT
**Source**: `trans_planckian/publications/mass-spec/harmonic-scattering-loops.tex`
**Key results**: rank(A)=C+1; sub-quadratic conditioning α=1.25;
  viscosity-refractive-index Spearman ρ=0.964 for H-bond liquids;
  18/18 benchmarks pass

### Paper 10: Superimposed Spectral Holograms and Three-State Completeness
**Role**: Superimposing ground/excited/emission spectra → spectral hologram
  H(ω,t) = Σ c_n(t)·S_n(ω)·exp(iφ_n(t)) encodes complete phase space.
  Emission events = natural timing triggers separating states with zero cross-talk.
  Six quantities inaccessible to single-state measurement:
  coupling matrix K_ij, Franck-Condon factors, Stokes shift decomposition,
  Huang-Rhys factors, Marcus reorganization energy, molecular symmetry.
  Autocatalytic SNR ∝ N^0.69 (beats √N). Empty dictionary principle.
  Validated: CH4+ (99.5% cross-prediction, V_ME=0.000) and Rhodamine 6G.
**Status**: PRESENT
**Source**: `chitungwiza/physics/superimposed-spectral-holograms.tex`
**Key results**: spectral hologram = complete categorical state function;
  emission lifetime τ_em = natural analog of LHC bunch crossing period τ_BX;
  hologram = trigger detector hit pattern H(channel,time)

### Paper 11: Measuring Atoms and Molecules Into Existence — Spectral Atlas
**Role**: Complete spectral atlas of H, H2, H2O via the categorical spectrometer.
  Two parallel derivation routes (Schrödinger and categorical spectrometer)
  converge on identical numerical values for 70 spectroscopic lines across 23 modalities.
  Four hardware oscillators (CPU clock→n, memory bus→l, LED→m, refresh→s) = CSCO.
  280/280 inter-modality agreement. Mean fractional error 0.40%.
  Empty dictionary principle validated.
**Status**: PRESENT
**Source**: `chitungwiza/physics/hyperfine-transition-spectra.tex`
**Key results**: categorical spectrometer measures atoms into existence;
  cross-modal commutation structurally guaranteed;
  0.001% error on atomic hydrogen spectra;
  9 decimal place agreement on 21cm hyperfine line

### Paper 12: Convertible Recursive Ensemble Strobes — Four-Tier Precision
**Role**: Four-tier precision refinement architecture for the categorical spectrometer.
  Tier 1 (single projection) → Tier 2 (ensemble loop, SNR∝N^0.67) →
  Tier 3 (recursive ternary depth, 3^d sub-projections) →
  Tier 4 (convertible strobes + Ritz combinations).
  Error: 0.0702% → 0.0065% → 0.0033% → 0.0018% across tiers.
  Ritz combinations (ω_AC = ω_AB + ω_BC) as internal self-consistency check.
  Triple convertibility: oscillation ↔ category ↔ partition round trips.
**Status**: PRESENT
**Source**: `trans_planckian/publications/mass-spec/convertible-recursive-spectra.tex`
**Key results**: four tiers = L1→HLT→offline→cross-subsystem at LHC;
  Ritz = energy-momentum conservation internal check;
  empty dictionary = structural incorruptibility

---

## VOLUME III — The LHC as a Partition Detector at TeV Scale
*Target audience: particle physicists, trigger engineers, HEP phenomenologists*
*Core claim: the LHC trigger is a partition spectrometer at TeV scale*

### Paper 13: Mathematical Foundations — Composition-Inflation and Planck Depth
**Role**: Composition-inflation for the trigger: T(n,d)=d(d+1)^{n-1} event categories.
  Planck depth n_P=60 for LHC (d=3, ν_B=40.079 MHz, derived).
  Temporal programming: ΔP(k) as sole datum, structural incorruptibility.
  Partition uncertainty ΔM·τ_p ≥ ℏ → 26 eV minimum per BX.
  c=Δx/τ_c → 25ns BX window = propagation time across 7.5m detector.
  Universal transport: all detector response times = partition lag τ_p=ℏ/ΔE.
**Status**: WORKING
**Source**: `accelerator/publications/trigger/paper-13-foundations.tex` (to write)
**Draws from**: CI, BPST, MTM, ION (S2-S5b of plan.md)

### Paper 14: Trigger as Backward Trajectory Completion
**Role**: S-entropy framework and receiver theory for the trigger.
  Triple Equivalence: Osc ≅ Cat ≅ Part applied to detector.
  Trigger as backward navigation O(log_3 N) in ternary hierarchy.
  Physical origin of cell registry from partition coordinates:
    tracker→n, ECAL→l, HCAL→m, muon→s, timing→parity.
  Selection rules as structural dead zones (type errors, not probabilistic suppression).
  QND multi-subsystem classification: all four subsystems commute.
  LHC as high-energy partition spectrometer: same Lagrangian as mass spec.
  Charge emergence→calorimeter; transport→detector response; C(n)→shell structure.
**Status**: WORKING (early draft)
**Source**: `accelerator/publications/trigger/paper-14-trajectory.tex` (to write)
**Draws from**: SR, SC, QPIO, ION, BPST (S6-S9c of plan.md)

### Paper 15: Virtual Particles as Virtual Sub-States and the Path Integral
**Role**: Five-part identification theorem: virtual particles = virtual sub-states.
  Off-shell (q²≠m²) ↔ outside [0,1]³.
  Feynman propagator ↔ mean-recovery constraint.
  Gauge invariance ↔ Local-Global Decoupling.
  S-matrix observability ↔ Path Opacity.
  Unitarity ↔ Cascade Power multiplicativity.
  Feynman diagrams as vaHera ASTs.
  Ward-Takahashi identities as Local-Global Decoupling theorem.
  Path integral as backward trajectory completion: O(log N) vs O(N) for classical.
  Feynman diagram count T(n,d) as composition-inflation lower bound.
**Status**: TODO
**Source**: `accelerator/publications/trigger/paper-15-virtual.tex` (to write)
**Draws from**: SC, SR, feynman1948path, ward1950identity (S10-S13 of plan.md)

### Paper 16: Constant Reduction and Three Routes to G for the LHC
**Role**: All seven SI constants in categorical angular units (Table with proof).
  Three routes to G applied to LHC: G determines t_P, t_P determines n_P=60.
  Three-route agreement at (d+1)^{-n} precision — LHC calibration test.
  HL-LHC dimension advantage: d=4 → n_P=52 (saves 8 crossings, 200ns headroom).
  Renormalization as cascade saturation: RG flow = federation composition.
  Dark sector as virtual-state-suppressed: Θ(N) vs O(log N) complexity.
  MOND scale a_0 = cH_0/(2π) from partition density.
  Dark energy w_eff = -0.75; secular dG/Gdt = -5.17×10^{-11} yr^{-1}.
**Status**: TODO
**Source**: `accelerator/publications/trigger/paper-16-constants.tex` (to write)
**Draws from**: CI, USD, BPST Part V (S14-S26 of plan.md)

### Paper 17: Validation and Unified Falsifiability Ledger
**Role**: Cross-domain validation spanning mass spectrometry → LHC.
  Three-route G test with LHC timing as calibration protocol.
  Selection-rule-violating backgrounds = zero (type errors, not rare).
  Partition extinction in SC magnets at 1.9K (R=0 exactly).
  Ritz combinations as trigger internal consistency check.
  Trigger performance metrics: η_C = 10% (90% unused categorical resolution).
  Falsifiability ledger: 10+ testable predictions with explicit falsification conditions.
  Cross-domain experimental validation chain:
    mass spec (NIST 4545 entries) → LHC (MIT-BIH cardiac analog) → TeV scale.
**Status**: TODO
**Source**: `accelerator/publications/trigger/paper-17-validation.tex` (to write)
**Draws from**: ION, QPIO, CRS, HTS, buhera B6 (S27-S30 of plan.md)

---

## VOLUME IV — The Buhera Runtime
*Target audience: systems software researchers, formal verification community*
*Core claim: Tempus programs compile to vaHera; selection rules become type errors*

### Paper 18: Tempus on Buhera — Formal Compiler and Bisimulation
**Role**: Tempus syntax → vaHera AST: total function (one-page compiler).
  Bisimulation theorem: Tempus operational semantics ≅ Buhera Kernel dispatch.
  Three LHC-specific kernel variants:
    StaticCellRegistryPve (registry immutability enforcement),
    IntervalTreePvCmm (timing cell lookup, ≤200ns latency),
    MonotoneCycleTem (M(t)/t = f_B to 10^{-12} over 10^9 BX).
  n_P-aware BuheraKernelBuilder with m_max_from_planck_depth(f,d).
**Status**: TODO (B4 from buhera-collaboration.md)
**Source**: `accelerator/publications/trigger/paper-18-tempus-on-buhera.tex` (to write)
**Draws from**: Buhera UTL, COE, IMP, KER Phase 1

### Paper 19: LHC Trigger as Buhera Application — Selection Rules as Type Errors
**Role**: Selection rules (Δl=±1, Δm∈{0,±1}) as vaHera typecheck rejection.
  Backgrounds requiring forbidden transitions = compile-time type errors.
  This is not probabilistic suppression — it is structural impossibility.
  Substrate requirements for Theorem 5.1 (structural incorruptibility) to hold
  in deployment (B5 from collaboration plan).
  Categorical utilization metric η_C as live kernel observable.
  PUI = n_operating/n_P = 100/60 = 1.67 (trigger above Planck depth).
**Status**: TODO (B3+B5 from buhera-collaboration.md)
**Source**: `accelerator/publications/trigger/paper-19-lhc-buhera.tex` (to write)
**Draws from**: Buhera KER, INT, B3, B5; BPST cons:selectionrules

### Paper 20: Conformance Suite — 10 LHC Tests + 30 Buhera Tests
**Role**: Complete validation suite with 40 automated tests.
  LHC subsuite (10 tests):
    validate_01_planck_depth.py (n_P=60 from Theorem S4)
    validate_02_registry_immutability.py (10^6 adversarial fragments, zero out-of-registry)
    validate_03_replay_rejection.py (10^4 replays, all rejected at δ≥w_min)
    validate_04_categorical_utilization.py (η_C matches log T(n_menu)/log T(n_P))
    validate_05_three_route_G.py (three routes agree at (d+1)^{-n})
    validate_06_selection_rule_rejection.py (forbidden Δl,Δm fail typecheck)
    validate_07_di_muon_pipeline.py (end-to-end di-muon trigger)
    validate_08_HL_LHC_d4.py (n_P=52 at d=4)
    validate_09_federation_at_nP.py (federation saturates at predicted asymptote)
    validate_10_cross_arch.py (V15-equivalent for LHC kernel build)
  Buhera subsuite: existing 30/30 PASS UTL+COE conformance tests.
**Status**: TODO (B6 from buhera-collaboration.md)
**Source**: `accelerator/publications/trigger/paper-20-conformance.tex` (to write)
**Draws from**: Buhera B6; all prior volumes for expected values

---

## PAPERS TO FETCH (copy into accelerator/sources/)

The following papers exist in other parts of the repository and need to be
copied into `accelerator/sources/` to be part of the monograph corpus.

### From trans_planckian/publications/sources/

```
bounded-phase-space-trajectory.tex     → Paper 1  (BPST, 476KB)
universal-shader-depth.tex             → Paper 2  (USD, G derivation)
ion-trajectory-completion-mechanism.tex→ Paper 3  (ION, partition Lagrangian)
mass-transfer-mechanism.tex            → Paper 8  (MTM, light/fluids/chromatography)
phase-locked-messaging.tex             → Appendix (PLM, secure communication)
phase-locked-live-streaming.tex        → Appendix (PLLS, streaming application)
phase-locked-finance.tex               → Appendix (PLF, settlement application)
cardiovascular-derivation.tex          → Appendix (cardiovascular partition)
cardio-neural.tex                      → Appendix (cardiac-neural coupling)
orthogonal-charge-quantification.tex   → Appendix (charge quantification)
euler-lagrangian.tex                   → Appendix (neural partition Lagrangian)
phase-space-mechanics.tex              → Appendix (categorical mechanics)
pyschon-circuit-mechanics.tex          → Appendix (psychon circuit mechanics)
```

### From trans_planckian/publications/mass-spec/

```
ion-thermodynamic-regimes.tex          → Paper 4  (five thermodynamic regimes)
categorical-state-counting.tex         → Paper 5  (fundamental identity)
mass-partitioning-computing.tex        → Paper 6  (mass computing)
quantupartite-ion-observatory.tex      → Paper 7  (single-ion, QND)
harmonic-scattering-loops.tex          → Paper 9  (light bending, transfer matrix)
convertible-recursive-spectra.tex      → Paper 12 (four-tier refinement)
```

### From trans_planckian/publications/upcoming-events/

```
trajectory-of-upcoming-events.tex      → Appendix (partition bijection, time succession,
                                          deletion argument, process argument, time travel)
```

### From chitungwiza/physics/

```
composition-inflation.tex              → Paper 2  (CI, T(n,d) formula)
superimposed-spectral-holograms.tex    → Paper 10 (spectral hologram)
hyperfine-transition-spectra.tex       → Paper 11 (atomic spectra atlas)
```

### From chitungwiza/epistemology/

```
unconstrained-subtask-recursion.tex    → Vol III companion (SR, S-entropy, miracle principle)
unconstrained-subtask-computing.tex    → Vol III companion (SC, backward trajectory)
```

### From Buhera repository (different repo — fetch separately)

```
long-grass/docs/os-throughput-law/universal-os-transport-law.tex
  → Paper 18 source (UTL, federation composition, five regimes)
long-grass/docs/computational-operations-equivalence/computational-operations-equivalence.tex
  → Paper 18 source (COE, three-route equivalence, time-count identity)
long-grass/implementation-plan.md
  → Paper 18 source (IMP, Rust kernel roadmap, Phase 1 shipped)
```

---

## PAPERS TO WRITE (new, in accelerator/publications/trigger/)

```
paper-13-foundations.tex    (S1-S5b: Planck depth, temporal programming)
paper-14-trajectory.tex     (S6-S9c: backward trajectory, cell registry, QND)
paper-15-virtual.tex        (S10-S13: virtual particles, Feynman, Ward, path integral)
paper-16-constants.tex      (S14-S26: constant reduction, G routes, cosmology)
paper-17-validation.tex     (S27-S30: falsifiability ledger, cross-domain validation)
paper-18-tempus-on-buhera.tex (Tempus compiler, bisimulation, LHC kernel variants)
paper-19-lhc-buhera.tex     (selection rules as type errors, substrate requirements)
paper-20-conformance.tex    (40 automated tests, 10 LHC + 30 Buhera)
```

The existing `lhc-trigger/lhc-trigger.tex` is the seed for Papers 13-17 (10 pages,
currently covering S1-S8, S10 partially, S14, S17, S18, S21, S23, S31).

---

## CROSS-REFERENCE MAP

The following table shows which papers cite which, to ensure consistency
when compiling the monograph. Each row is a paper; each column is a paper
it draws from directly.

```
Paper | Draws from
------+-----------
  2   | 1 (BPST Part V)
  3   | 1 (BPST), 2 (CI)
  4   | 3 (ION), 1 (BPST)
  5   | 3 (ION), 1 (BPST)
  6   | 5, 2 (CI), 3 (ION)
  7   | 3 (ION), 1 (BPST)
  8   | 1 (BPST), 3 (ION)
  9   | 8 (MTM), 1 (BPST), 7 (QPIO)
 10   | 9, 7 (QPIO), 1 (BPST)
 11   | 1 (BPST), 7 (QPIO), 10
 12   | 11, 7 (QPIO), 9, 1 (BPST)
 13   | 2 (CI), 1 (BPST), 8 (MTM), 7 (QPIO)
 14   | SC, SR, 7 (QPIO), 3 (ION), 1 (BPST), 9 (HSL)
 15   | SC, SR, feynman1948path, ward1950identity
 16   | 2 (CI), USD, 1 (BPST Part V)
 17   | 3 (ION), 7 (QPIO), 12 (CRS), 11 (HTS), Buhera B6
 18   | Buhera UTL, COE, IMP, KER; Tempus paper
 19   | 18, 1 (BPST cons:selectionrules), Buhera B3 B5
 20   | All papers 1-19 (for expected values)
```

---

## SHARED THEOREMS (appear in multiple papers, need consistent numbering)

These are the same theorem stated in different frameworks.
When compiling, cross-cite each occurrence to the canonical statement.

| Theorem | Canonical paper | Also appears in |
|---|---|---|
| C(n) = 2n² | Paper 1 (BPST §3) | Papers 3,4,5,6,7,9,11,12,13,14 |
| T(n,d) = d(d+1)^{n-1} | Paper 2 (CI Thm 3.2) | Papers 6,13,14,16,17 |
| dM/dt = 1/⟨τ_p⟩ | Paper 5 (Thm 2.1) | Papers 3,7,12,13 |
| E = ℏω | Paper 1 (BPST cons:energyfrequency) | Papers 3,7,8,11 |
| E = mc² | Paper 1 (BPST cons:emc2) | Papers 3,14 |
| c = 2π rad/tick | Paper 2 (CI §8) | Papers 8,13,16 |
| k_B = E_tick/ln(d+1) | Paper 2 (CI §8) | Papers 13,16 |
| Triple Equivalence (Osc≅Cat≅Part) | Paper 1 (BPST §5) | Papers 3,5,6,7,11,12,13,14 |
| Structural incorruptibility | Paper 13 (Thm S5.1) | Papers 14,19 |
| Virtual particle identification | Paper 15 (Thm main) | Papers 14 |
| Selection rules as type errors | Paper 19 (main result) | Papers 14,17 |
| Three routes to G convergence | Papers 1+2 (BPST+CI) | Papers 16,17 |
| PV = k_B T_cat for N=1 | Paper 7 (QPIO Thm 6) | Papers 13 |
| Rank(A) = C+1 | Paper 9 (HSL Thm main) | Papers 10,13,14 |

---

## APPENDIX PAPERS

These papers are important supporting material but do not need
to be one of the 20 main papers. They should be included as appendices
or cited as companion papers.

### Partition Bijection and the Arrow of Time
**File**: `trans_planckian/publications/upcoming-events/trajectory-of-upcoming-events.tex`
**Role**: Proves the partition bijection Φ: Z+ → P; strict monotonicity;
  incoherence of decrement; successor existence; cyclic closure.
  The four converging arguments against time reversal:
  formal, observational, deletion, process.
  Validated on 6855 Orbitrap ions.
**Why appendix**: This is the "pure physics" version of the fundamental identity;
  the main papers (5, 13) cite its results but the full proofs belong here.

### Phase-Locked Messaging and Applications
**Files**: `phase-locked-messaging.tex`, `phase-locked-live-streaming.tex`, `phase-locked-finance.tex`
**Role**: Three application papers showing PLM applied to secure communication,
  streaming, and financial settlement. Support the PLM section of Paper 14.
**Why appendix**: Applications of the trigger's communication model, not core physics.

### Neuroscience and Cardiology Papers
**Files**: `cardio-neural.tex`, `cardiovascular-derivation.tex`,
  `orthogonal-charge-quantification.tex`, `euler-lagrangian.tex`,
  `phase-space-mechanics.tex`, `pyschon-circuit-mechanics.tex`
**Role**: Validate the partition framework across biological scales:
  cardiac mechanics, neural oscillations, charge quantification, consciousness.
  Support the multi-scale validation claims in Paper 17.
**Why appendix**: Cross-domain validation material, not primary LHC physics.

---

## WRITING PRIORITY ORDER

Based on dependencies, write new papers in this order:

1. Paper 13 (foundations) — seed already exists in lhc-trigger.tex; extend S1-S5b
2. Paper 14 (trajectory+cell registry) — key new contribution; uses BPST, SC, ION, QPIO
3. Paper 15 (virtual particles) — uses SC, SR; extends existing §10-13 in lhc-trigger.tex
4. Paper 16 (constants+cosmology) — uses CI, USD; extends existing §14-26
5. Paper 17 (validation) — depends on all prior; extends existing §27-30
6. Paper 18 (Tempus on Buhera) — needs Buhera repo access
7. Paper 19 (LHC Buhera app) — depends on Paper 18
8. Paper 20 (conformance suite) — depends on Papers 13-19

---

## NOTES ON PAPER STRUCTURE

Each paper in the monograph should follow the same format:
- RevTeX4-2 or article class (match existing papers)
- Abstract citing which other volumes it connects to
- Introduction stating its role in the monograph
- Main body with complete proofs
- Experimental validation section
- Cross-volume outlook section connecting to the next volume

Papers 1-12 already exist and should be copied as-is.
Papers 13-20 need to be written following the plan.md section plan.

The existing `lhc-trigger/lhc-trigger.tex` is the seed for Papers 13-17.
It currently covers: S1-S8 (foundations, trigger, trajectory), S10 (virtual, partial),
S14 (constants, partial), S17-S18 (ATLAS analysis, partial), S21 (RG, partial),
S23 (dark sector, partial), S31 (conclusion).

---

*Last updated: based on full corpus discussion*
*Total papers: 20 main + ~10 appendices*
*Total source papers fetched: 20+ from trans_planckian, chitungwiza, and Buhera repos*
