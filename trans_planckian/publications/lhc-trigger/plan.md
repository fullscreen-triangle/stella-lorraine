# LHC Trigger Paper: Full Section Plan

**Title**: On the Categorical Mechanics of Real-Time Event Selection in Bounded Phase Space:
Composition-Inflation, Temporal Programming, and S-Entropy Trajectory Completion in LHC Trigger Systems

**Status key**: DONE | EXTEND | TODO

**Source papers**:
- BPST = bounded-phase-space-trajectory.tex (master deductive framework)
- USD  = universal-shader-depth.tex (three routes to G)
- CI   = composition-inflation.tex (T(n,d) formula, angular resolution)
- SR   = unconstrained-subtask-recursion.tex (S-entropy, miracle principle)
- SC   = unconstrained-subtask-computing.tex (backward trajectory, virtual sub-states)
- ION  = ion-trajectory-completion-mechanism.tex (partition Lagrangian, mass spec, mass from 3 routes)
- MTM  = mass-transfer-mechanism.tex (light from partition: c=Dx/tau_c; universal transport mu=tau_c*g)
- QPIO = quantupartite-ion-observatory.tex (single-ion ideal gas PV=kT_cat; complete commutation; QND)
- HSL  = harmonic-scattering-loops.tex (light bending: rank(A)=C+1; one ray -> C+1 sources; N_max=(C+1)*T_deph/T_L)
- SSH  = superimposed-spectral-holograms.tex (spectral hologram; three-state superposition; one beam = complete scene)
- HTS  = hyperfine-transition-spectra.tex (categorical spectrometer on H/H2/H2O; empty dictionary; 280/280)
- CRS  = convertible-recursive-spectra.tex (four-tier refinement; Ritz combinations; 0.07%->0.003%)

---

## PART I — MATHEMATICAL FOUNDATIONS

### S1 Introduction [EXTEND]
- The event selection problem: 40 MHz, 99.997% rejection
- Three structural properties: irreversibility, timing primacy, exponential state space
- Overview of all results across all parts
- Needs: full table of contents; forward refs to all parts

### S2 Bounded Phase Space and Partition Algebra [DONE]
- Axiom of Bounded Phase Space (from BPST)
- Partition Existence Theorem
- Poincare Recurrence; Oscillatory Necessity Corollary
- Partition coordinates (n,l,m,s); Shell capacity C(n) = 2n^2
- Source: BPST Part I Consequences 1-7

### S3 Composition-Inflation Mechanism [DONE]
- Integer compositions count: 2^{n-1}
- Labeled compositions in d dimensions
- Central formula: T(n,d) = d(d+1)^{n-1}
- Growth properties; angular resolution; no Planck limit on dimensionless resolution
- Source: CI paper Theorem 3.2; BPST Consequence cons:compositioninflation

---

## PART II — PLANCK DEPTH AND TRIGGER DESIGN

### S4 The Planck Depth for LHC Trigger Systems [DONE]
- Planck depth definition and formula n_P = 1 + ceil(log_{d+1}(tau/d*t_P))
- LHC Planck depth: n_P = 60 for d=3, nu_B = 40.079 MHz
- Table: Planck depths across oscillators
- Corollary: Universality n_P in [36,73]
- HL-LHC advantage: d=4 reduces n_P from 60 to 52

### S5 Temporal Programming and Structural Incorruptibility [EXTEND]
- DeltaP(k) = T_ref(k) - t_rec(k) as sole runtime datum
- Timing cells; cell registry Gamma; temporal trigger program definition
- Theorem: Structural incorruptibility of timing-only triggers
- Needs: formal Tempus program syntax for di-muon trigger
- Needs: proof that zero injection rate follows from incorruptibility

### S5b Partition Uncertainty, Speed of Light, and Fundamental Timing [TODO]
- FROM ION PAPER: fundamental identity dM/dt = 1/<tau_p> (state counting is intrinsically digital)
- Partition uncertainty theorem: DeltaM * tau_p >= hbar
- Applied to LHC bunch crossing (tau_p = 25 ns):
    DeltaM >= hbar/tau_p = 1.055e-34 / 25e-9 = 4.2e-27 J ~ 26 eV
    This is the minimum detectable partition depth change per bunch crossing
- This is not an engineering constraint -- it is a theorem of the partition framework

- FROM MTM PAPER: c = Delta_x / tau_c (speed of light from partition propagation)
- Applied to ATLAS interaction region (Delta_x = 7.5 m, detector half-diameter):
    tau_c = Delta_x / c = 7.5 m / (3e8 m/s) = 25 ns = EXACTLY THE LHC BUNCH CROSSING PERIOD
- KEY RESULT: The 25 ns bunch crossing window is NOT an engineering choice --
  it is the partition propagation time for the ATLAS interaction region
  tau_BX = c^{-1} * Delta_x_detector
- Cell half-width |DeltaP| < 12.5 ns = tau_c/2: partition uncertainty half-period

- Universal transport from MTM: tau_p = hbar/DeltaE unifies ALL detector response times
    Silicon tracker: tau_c = hbar/E_gap ~ 10 fs -> carrier drift time
    ECAL crystals (PbWO4): tau_c = hbar/E_gamma ~ 1 fs -> scintillation rise time
    Liquid argon: mu = tau_c * g -> drift velocity in electric field
    Muon drift tubes: tau_c from gas molecule partition lag -> 20 ns signal timing
- Theorem: Every detector timing parameter is the partition lag tau_c for that material
- The Lorentz force law F = q(E + v x B) is the Euler-Lagrange equation of the
  partition Lagrangian (from ION paper) -- derived, not assumed
- Sources: ION paper S9; MTM paper S3-4

---

## PART III — S-ENTROPY FRAMEWORK AND TRAJECTORY COMPLETION

### S6 The S-Entropy Framework and Receiver Theory [TODO]
- Bounded receiver: (Sigma, Phi, K, dec)
- S-functional axioms S1-S4; S-scale [0,100]
- Floor Positivity Theorem: S_flat(R) > 0 for every bounded receiver
- Information Bound Theorem; Floor as dual to address space
- Source: SR paper S1-2; SC paper S1-2

### S7 Triple Equivalence: Oscillatory, Categorical, Partition [TODO]
- Three algebraic structures: Osc, Cat, Part
- Conversion functors F_OC, F_CP, F_PO (explicit constructions)
- Triple Equivalence Theorem (categorical equivalences)
- Corollary: Free Conversion; Optimal Representation Theorem
- Source: SR paper S5; SC paper S2

### S8 The Trigger as Backward Trajectory Completion [EXTEND]
- Ternary refinement hierarchy; S-entropy embedding into [0,1]^3
- Trigger State Inflation: T(n,d) = d(d+1)^{n-1}
- Backward Navigation O(log_3 N); Collapse without virtual states Theta(N)
- Needs: explicit worked LHC event example
- Needs: proof that detector hierarchy matches ternary hierarchy

### S9 Physical Origin of Trigger Cell Registry from Partition Coordinates [TODO]
- Partition coordinates (n,l,m,s) from BPST derive five quantum numbers
- Map to five detector subsystems:
    inner tracker <-> principal n
    EM calorimeter <-> angular l
    hadronic calorimeter <-> projection m
    muon system <-> chirality s
    timing (HL-LHC) <-> parity / fifth dimension
- Selection rules (Delta-l = +/-1, Delta-m in {0,+/-1}) as trigger dead zones:
    transitions violating selection rules produce no detector signal
    backgrounds structurally excluded, not probabilistically suppressed
- Charge emergence as partition boundary -> calorimeter operating principle
- Transport coefficients from partition lag -> detector response times
    Drude (resistivity), Chapman-Enskog (viscosity), Einstein (diffusion) as special cases
    timing window |DeltaP| < 12.5 ns is transport-determined, not engineered
- Partition extinction -> superconducting magnets at 1.9 K (R=0 exactly)
- Source: BPST Consequences: quantumnumbers, selectionrules, chargeemergence,
  transport, partitionextinction

### S9c Multi-Subsystem QND Classification [TODO]
- From QPIO paper: all partition coordinates commute:
    [n_hat, l_hat] = [l_hat, m_hat] = [m_hat, s_hat] = ... = 0
- QND (Quantum Non-Demolition) property: [O_hat, H_hat] = 0 for all partition coordinates
    => measuring n in tracker does not disturb l in ECAL
    => measuring l in ECAL does not disturb m in HCAL
    => all four commuting observables determined simultaneously without backaction
- Backaction bound: Delta_p/p <= lambda_dB/(4*pi*L) ~ 10^{-11}
    Three orders below thermal fluctuations -- QND property is practically perfect
- Map to LHC detector subsystems:
    Tracker   -> n (principal, momentum/energy scale, cyclotron frequency)
    ECAL      -> l (angular complexity, shower shape)
    HCAL      -> m (orientation, hadronic fraction)
    Muon      -> s (chirality, charge sign)
    Timing    -> 5th coordinate (parity/phase)
- Theorem: Multi-Subsystem QND Trigger
    The LHC trigger determines (n,l,m,s) of each detected particle from one detector pass
    No sequential pipeline ordering required: all subsystems fire simultaneously
    This is why L1 can combine ECAL + HCAL + muon in one latency window (2.5 mus)
- Categorical temperature of trigger:
    T_cat = hbar * omega_BX / (2*pi * k_B) = hbar * 40MHz / (2*pi * k_B) ~ 31 nK
    hbar/(k_B * T_cat) = 1/omega_BX = tau_BX = 25 ns (consistency check)
- Single-particle ideal gas for LHC:
    Each collision product satisfies PV = k_B * T_cat (N=1 limit)
    T_cat = hbar * omega_c / (2*pi * k_B) for particle in solenoid field
    omega_c = qB/m (cyclotron frequency in tracker B=2T)
- Source: QPIO paper S3 (commutation), S4 (thermodynamics), S5 (triple equivalence)

### S9b The LHC as a High-Energy Partition Spectrometer [TODO]
- Core thesis: the LHC detector IS a mass spectrometer at TeV energy scale
- The partition Lagrangian (from ION paper) governs both instruments:
    L_M = (1/2)*mu*|xdot|^2 + mu*xdot*A_M - M(x,t)
    where mu = alpha*(m/z) is partition inertia
- Lorentz force F = q(E + v x B) is the Euler-Lagrange equation of L_M (DERIVED not assumed)
- Four analyzer types -> four LHC detector technologies:
    TOF (T ~ sqrt(m/z))         -> timing detectors (HGTD, ETL)
    Quadrupole (Mathieu)        -> dipole magnets + tracker (bending in B field)
    Orbitrap (omega ~ sqrt(z/m))-> calorimeter energy measurement
    FT-ICR (omega_c ~ z/m)     -> muon spectrometer (cyclotron in solenoid)
- Collision products as partition malformations:
    Higgs boson, top quark, W/Z are incomplete categorical structures
    They minimize partition depth by decaying
    The trigger reads the memory of the collision's partition history
- Ions as partition malformations (ION paper):
    M_ion = kB*T*ln(Z!/(Z-z)!) ~ z*kB*T*ln(Z)
    Exactly the same structure as M_collision ~ z_eff * ln(E/E_0)
- Theorem: The partition Lagrangian with M_field(x) = -kappa*z (linear)
    gives T ~ sqrt(m/z) (TOF equation) -- identical for both MS and LHC timing
- Experimental validation link:
    ION paper validated on 4545 NIST entries + 127000 proteomic bijections
    Same partition Lagrangian governing LHC detector physics
    Cross-domain validation: MS at keV scale -> LHC at TeV scale, same framework
- Source: ION paper S7-9 (ion, Lagrangian, analyzers)

---

## PART IV — VIRTUAL PARTICLES AND QUANTUM FIELD THEORY

### S10 Virtual Sub-States and Virtual Particles [EXTEND]
- QFT review: off-shell q^2 != m^2c^4; Feynman propagator D_F(q)
- Five-Part Identification Theorem (virtual particles = virtual sub-states)
- Needs: formal proofs of items (3) gauge invariance and (4) path opacity

### S11 Feynman Diagrams as vaHera Expressions [TODO]
- vaHera AST: Literal, Call, Compose, Hole
- External particles = Literals; vertices = Calls; propagators = Compose; virtuals = sub-states
- Compositionality Lemma
- Unconstrained Subtask Theorem applied to Feynman amplitudes
- Miracle Principle = perturbation theory foundation
    each diagram gauge-dependent (local S=100), sum physical (global S*)
- Feynman diagram count T(n,d) as lower bound at order n
- Dyson non-convergence as cascade non-saturation
- Source: SC Part III-IV; SR paper S3-4

### S12 Ward-Takahashi as Local-Global Decoupling [EXTEND]
- Ward identity: q^mu Gamma_mu(p,p') = S_F^{-1}(p') - S_F^{-1}(p)
- Theorem already stated; needs formal proof
- Gauge parameter xi = free local S-value assignment
- 4-momentum conservation = mean-recovery constraint

### S13 The Path Integral as Backward Trajectory Completion [TODO]
- Feynman path integral as sum over backward trajectories
- Physical amplitude = global S-value; each path = labeled composition
- Stationary phase = backward navigation O(log N) complexity
- Forward path integral (constructive QFT) = O(N) complexity
- Theorem: Virtual paths necessary for sub-exponential path integral evaluation
- Remark: Lattice QCD as truncated path sum at depth n_P
- Source: feynman1948path; SC paper Part IV

---

## PART V — ANGULAR REFORMULATION OF FUNDAMENTAL CONSTANTS

### S14 Constants Reduction to Categorical Angular Units [EXTEND]
- c = 2pi rad/tick; hbar = E_tick/(2pi); k_B = E_tick/ln(d+1)
- m_0 = E_tick*nu_0/(4pi^2); E_0 = E_tick*nu_0
- t_P = sqrt(E_tick*G)/(2pi)^3
- G is the unique irreducible constant
- Needs: rigorous proof that k_B = E_tick/ln(d+1) from composition entropy

### S15 Three Routes to the Gravitational Constant [TODO]
- Route I: oscillation-ratio G^(I) = (c^3/hbar)*A_12^2*pi
- Route II: category fixed-point G^(II) via fixed-point equation g* = 1/(1+27^{-56})
- Route III: partition-density ratio G^(III) = (c^3/hbar)*pi*R^{1/(d+1)}*T_ref^{-1}
- Three-route convergence theorem: |G^(i) - G^(j)| <= K*(d+1)^{-n}
- Framework closure corollary: no physical constant exterior to counting
- Numerical table: Routes at n=8,15,27,56 vs CODATA
- Source: USD S3-4; BPST Part V

### S16 Planck Depth Precision from G-Derivation Routes [TODO]
- n_P = 60 depends on t_P which depends on G (CODATA uncertainty ~2.2e-5)
- G-routes surpass CODATA precision at n >= 8 cycles
- Theorem: n_P computable to (d+1)^{-n} precision via three routes
- Three-route n_P values all give 60 (integer, routes agree)
- Trigger calibration at n >= 60 is G-precision-limited, not timing-limited
- Three-route G measurement as a trigger calibration check
- Source: USD S5; CI S5

---

## PART VI — LHC TRIGGER ANALYSIS

### S17 Current LHC Trigger in Composition-Inflation Terms [EXTEND]
- L1 trigger: n=100 crossings, n/n_P = 1.67
- Menu ~10^3 paths -> composition depth n~6 (10% of n_P)
- Unused categorical resolution: ~90%
- Needs: quantitative analysis of exploiting more available resolution

### S18 HL-LHC Upgrade [EXTEND]
- d=4 gives n_P = 52 (theorem)
- HGTD precision timing 20-30 ps
- T(52,4) = 4*5^{51} ~ 2.8e35
- Needs: actual HGTD timing resolution and cell width analysis

### S19 Phase-Locked Communication and Detector Synchronization [TODO]
- LHC bunch clock as T_ref (40.079 MHz)
- Phase coherence between subsystems = coherent regime (R_c >= 0.95)
- Kuramoto order parameter R_c = exp(-2pi^2*CV^2) from inter-BX timing variance
- Regime classification for detector timing (turbulent/aperture/cascade/coherent/phase-locked)
- Double helix model: A-side and C-side as complementary strands
- CSPLM: each arm observes complement of other's signal
- Theorem: MITM immunity of timing trigger
- Consciousness window formula applied to trigger window: Delta_t_C = T_BX/(2pi*sqrt(R_A*R_B))
- Source: PLM paper; cardio-neural paper; BPST

---

## PART VII — RENORMALIZATION AND CASCADE CATALYSIS

### S20 Multiplicative Catalyst Algebra [TODO]
- Catalyst definition; catalytic power kappa(gamma)
- Multiplicativity: kappa(gamma1 diamond gamma2) = 1 - (1-kappa1)(1-kappa2)
- Cascade Power corollary
- Geometric Decay theorem
- Cascade Saturation theorem: saturates iff sum(kappa_i) = infinity
- Source: SR S5-6; SC Part V

### S21 Renormalization as Cascade Saturation [EXTEND]
- Already stated; needs connection to Callan-Symanzik equation
- Wilson RG = cascade of virtual loop corrections
- Fixed point g* = saturation point
- Asymptotic freedom = saturation to trivial fixed point

### S22 Renormalization Group in Composition-Inflation Language [TODO]
- Running coupling at depth n: g(n) = S-value at scale n
- Beta function: dg/d(log mu) = -dkappa/dn
- Relevant operators at mass scale mu: T(n_mu, d) where n_mu = log_{d+1}(Lambda/mu)
- Lattice QCD cutoff = composition depth n_P for strong coupling
- Theorem: number of independent renormalization parameters = d (one per entropy dimension)
- Source: wilson1971renormalization; callan1970broken; CI paper

---

## PART VIII — DARK SECTOR AND COSMOLOGICAL CONSEQUENCES

### S23 Dark Matter as Virtual-State-Suppressed Sector [EXTEND]
- Already stated; needs connection to observed dark matter density
- Suppression ratio ~N/log(N) ~ 10^{78}
- Gravitational coupling through G only (irreducible)

### S24 MOND Scale from Partition Density [TODO]
- G_eff(r) = G_0[1 + alpha/(r/r_0)^{1/4}] in low-density regime
- Characteristic acceleration a_0 = cH_0/(2pi) ~ 1.04e-10 m/s^2
- Comparison to McGaugh-Milgrom observations (13% off)
- Source: USD S6; BPST Part V

### S25 Dark Energy Equation of State w = -0.75 [TODO]
- Partition density decreases as universe expands
- dG/G = -3H/(d+1) (secular drift)
- Theorem: w_eff = -1 + 1/(d+1) = -0.75 constant with redshift
- Comparison: Pantheon w_obs = -1.03 +/- 0.10 (within 2.5 sigma)
- Source: USD S6; BPST Part V

### S26 Secular Drift of G [TODO]
- dG/Gdt = -3H_0/(d+1) ~ -5.17e-11 yr^{-1}
- LLR bound: |dG/Gdt| < 1e-12 yr^{-1}
- One decade above current bounds; falsifiable by next-decade LLR
- LHC luminosity calibration as short-timescale test
- Source: USD S6

---

## PART IX — EXPERIMENTAL VALIDATION AND FALSIFIABILITY

### S27 Three-Route G Test with LHC Timing [TODO]
- Protocol: 56 caesium cycles, compute G via each route, compare to CODATA
- LHC timing version: compute n_P from measured timing precision
- Three-route disagreement > (d+1)^{-n} signals calibration error
- Theorem: Three-route agreement iff framework consistent

### S28 Trigger Performance Metrics [TODO]
- Composition Efficiency: eta_C = log T(n_menu,d) / log T(n_P,d)
- Current ATLAS: n_menu~6, n_P=60, eta_C ~ 10%
- Planck Utilisation Index: PUI = n_operating/n_P = 100/60 = 1.67
- Background rejection = fraction failing cell conditions

### S29 Selection-Rule-Violating Backgrounds [TODO]
- Selection rules as structural dead zones (from S9)
- Prediction: backgrounds requiring rule-violating intermediates = zero categorically
- Not probabilistic suppression -- logical exclusion
- Falsification: any confirmed background event violating selection rules

### S19b Light, Fluids, and Chromatography as Partition Geometry [TODO]
- The three phenomena unified by MTM paper directly map to LHC detector components:
    Chromatography (t_R = (L/u_0)*S) -> TOF detectors (T ~ sqrt(m/z))
    Fluid viscosity (mu = tau_c * g) -> liquid argon calorimeter, gas drift chambers
    Light as mediator (c = Dx/tau_c) -> EM calorimeter, photodetectors, Cherenkov detectors
- Self-consistency loop:
    Light (from partition geometry) validates partition coordinates (via UV-Vis spectroscopy)
    Partition coordinates determine retention times (chromatography)
    Retention times and viscosity share tau_c parameter
    All three validated against experiment: 2%, 3.2%, 1-6% errors
- At the LHC: the same self-consistency loop operates
    ECAL photon energy E = hbar*omega validates particle identities (partition coordinates)
    Timing measurements determine t_R (trajectory traversal times)
    Calorimeter fluid response governed by mu = tau_c * g
- Theorem: c = Dx/tau_c -> tau_BX = Delta_x_ATLAS / c = 25 ns (derived, not input)
- This is the key result: the LHC trigger timing is derived from c and detector geometry,
  both of which are consequences of the partition framework
- Optical-mechanical partition lag ratio: tau_c^{opt}/tau_c^{mech} ~ 2.0 (from MTM paper)
  -> ratio of optical (ECAL) to mechanical (tracker) timing resolutions at ATLAS
- Source: MTM paper S3-4; ION paper S8

### S29b Cross-Domain Validation via Mass Spectrometry [TODO]
- Same partition Lagrangian governs both mass spectrometers and LHC detectors
- ION paper validates on 4545 NIST library entries (100% conformance)
- ION paper validates on 127,000 proteomic bijective transformations
- Additional: trajectory-of-upcoming-events paper validates on 6855 Orbitrap ions
  (mean partition count Mbar = 2,013,006, CV < 10^{-6})
- Prediction: partition inertia mu = alpha*(m/z) should be measurable at the LHC
  for each particle type via its response to electromagnetic fields
- Map: ion m/z ratio <-> LHC particle species (electron, muon, pion, kaon, proton)
- Prediction: T ~ sqrt(E/z) for timing detectors (same form as TOF equation)
  where E is particle kinetic energy and z is electric charge
- This is a new falsifiable prediction derivable from the partition Lagrangian
- Falsification: if timing detector response deviates from T ~ sqrt(E/z) form,
  the partition Lagrangian is wrong
- Source: ION paper S13 (validation); trajectory-of-upcoming-events S9

### S30 Unified Falsifiability Ledger [TODO]
- Complete table of predictions, values, status, and tests
- Entries include: n_P=60, HL-LHC n_P=52, three-route G agreement, zero rule-violation backgrounds,
  partition extinction in SC magnets, dark sector null result, w=-0.75, dG/Gdt, MOND a_0, eta_C<10%

---

## CONCLUSION AND APPENDICES

### S31 Conclusion [EXTEND]
- Summary of results
- Needs: explicit statement of three-paper collaboration
- Needs: connection back to single axiom

### Appendix A — Deductive Flow Diagram [TODO]
- Chain: Axiom -> Oscillatory Necessity -> Partition Coordinates -> C(n)=2n^2
  -> Composition-Inflation -> Planck Depth -> Trigger State Space
  -> Backward Trajectory -> Structural Incorruptibility -> LHC Design

### Appendix B — Numerical Tables [TODO]
- Planck depth table: d=1..6, representative oscillators
- T(n,d) for n=1..100, d=1..5
- Three-route G values at n=8,15,27,56 (from USD)

### Appendix C — Glossary [TODO]
- DeltaP(k), composition-inflation, Planck depth, virtual sub-state,
  categorical aperture, S-floor, structural incorruptibility, triple equivalence

---

## Writing Order

Write sections in this dependency order:
1.  S6   (S-entropy framework) -- needed by S7, S8, S11, S13
2.  S7   (triple equivalence) -- needed by S8
3.  S5b  (partition uncertainty + c = Dx/tau_c) -- uses ION + MTM; quick win, high impact
4.  S9   (partition coordinates -> cell registry) -- key contribution, uses BPST
5.  S9b  (LHC as partition spectrometer) -- uses ION paper, extends S9
6.  S11  (Feynman diagrams as vaHera) -- extends S10
7.  S13  (path integral as backward completion)
8.  S15  (three routes to G) -- needed by S16
9.  S16  (n_P precision from G routes) -- six convergent derivations
10. S19  (PLM and detector synchronization)
11. S19b (light, fluids, chromatography as partition geometry) -- uses MTM
12. S20  (catalyst algebra) -- needed by S22
13. S22  (RG in composition-inflation)
14. S24-S26 (cosmological consequences)
15. S27-S30 (experimental validation and falsifiability)
16. Appendices

---

## Current Status

- Pages written: 10 (lhc-trigger.tex)
- Sections DONE: S2, S3, S4
- Sections EXTEND: S1, S5, S8, S10, S12, S14, S17, S18, S21, S23, S31
- Sections TODO: S5b, S6, S7, S9, S9b, S11, S13, S15, S16, S19, S20, S22, S24-S30, Appendices
- Target: 110-130 pages two-column (increased by ION + MTM + QPIO paper contributions)

## Monograph Structure (20 papers across 5 volumes)

### Volume I -- Mathematical Physics Foundation
- Paper 1 (BPST): Single axiom -- all physics consequences
- Paper 2 (CI + USD): Composition-inflation + three routes to G

### Volume II -- Partition Spectroscopy at Molecular Scale
- Paper 3 (ION): Partition Lagrangian; mass as partition memory
- Paper 4 (Regimes): Five thermodynamic regimes; universal EOS; 5277 ions
- Paper 5 (Counting): dM/dt = 1/<tau_p>; heat-entropy decoupled; irreversibility
- Paper 6 (Computing): Ternary addresses; partition determinism; 96.3% accuracy
- Paper 7 (QPIO): Single-ion ideal gas PV=k_B*T_cat; complete commutation; QND
- Paper 8 (MTM): Light as partition mediator c=Dx/tau_c; universal transport

### Volume IIb -- Categorical Optics: One Beam, Complete Scene
- Paper 9  (HSL):  Harmonic-scattering loops; transfer matrix rank = C+1; one ray -> C+1 sources; benzene validated
- Paper 10 (SSH):  Spectral hologram; three-state superposition; H(w,t)=sum c_n S_n exp(i*phi_n); six new quantities
- Paper 11 (HTS):  Categorical spectrometer on H/H2/H2O; 280/280 agreement; empty dictionary principle
- Paper 12 (CRS):  Four-tier refinement; Ritz combinations as internal check; 0.07%->0.003% precision

### Volume III -- LHC as Partition Detector at TeV Scale
- Paper 13: Mathematical foundations S1-S5b (Planck depth, temporal programming, harmonic detector coupling)
- Paper 14: Trigger as backward trajectory completion S6-S9c (S-entropy, cell registry, QND, four-tier)
- Paper 15: Virtual particles and QFT S10-S13 (Feynman diagrams, Ward identities)
- Paper 16: Constants and cosmology S14-S26 (G from three routes, dark sector, MOND)
- Paper 17: Validation and falsifiability S27-S30 (cross-domain, Ritz combinations, falsifiability ledger)

### Volume IV -- Buhera Runtime
- Paper 18 (B4): Tempus on Buhera -- formal compiler and bisimulation theorem
- Paper 19 (B3+B5): LHC trigger as Buhera application -- selection rules as type errors
- Paper 20 (B6): Conformance suite -- 10 LHC tests + 30 Buhera tests

## Source Paper Contribution Map (updated)

| Section | BPST | USD | CI | SR | SC | ION | MTM | QPIO |
|---|---|---|---|---|---|---|---|---|
| S2  | primary | -- | -- | -- | -- | -- | -- | -- |
| S3  | partial | -- | primary | -- | -- | -- | -- | -- |
| S4  | -- | partial | primary | -- | -- | -- | -- | -- |
| S5b | -- | -- | -- | -- | -- | primary | primary | primary |
| S6  | -- | -- | -- | primary | partial | -- | -- | -- |
| S7  | -- | -- | -- | primary | partial | -- | -- | partial |
| S8  | partial | -- | partial | -- | primary | -- | -- | -- |
| S9  | primary | -- | -- | -- | -- | partial | -- | partial |
| S9b | partial | -- | -- | -- | -- | primary | partial | partial |
| S9c | -- | -- | -- | -- | -- | -- | -- | primary |
| S10 | -- | -- | -- | partial | primary | -- | -- | -- |
| S11 | -- | -- | -- | primary | primary | -- | -- | -- |
| S14 | primary | primary | primary | -- | -- | primary | primary | primary |
| S15 | partial | primary | -- | -- | -- | partial | -- | -- |
| S19 | partial | -- | -- | -- | -- | -- | partial | -- |

## Source Paper Contribution Map

| Section | BPST | USD | CI | SR | SC | ION |
|---|---|---|---|---|---|---|
| S2  | primary | -- | -- | -- | -- | -- |
| S3  | partial | -- | primary | -- | -- | -- |
| S4  | -- | partial | primary | -- | -- | -- |
| S5  | -- | -- | -- | -- | -- | -- |
| S5b | -- | -- | -- | -- | -- | primary |
| S6  | -- | -- | -- | primary | partial | -- |
| S7  | -- | -- | -- | primary | partial | -- |
| S8  | partial | -- | partial | -- | primary | -- |
| S9  | primary | -- | -- | -- | -- | partial |
| S9b | partial | -- | -- | -- | -- | primary |
| S10 | -- | -- | -- | partial | primary | -- |
| S11 | -- | -- | -- | primary | primary | -- |
| S13 | -- | -- | -- | -- | primary | -- |
| S14 | primary | primary | primary | -- | -- | primary |
| S15 | partial | primary | -- | -- | -- | partial |
| S16 | -- | primary | primary | -- | -- | -- |
| S19 | partial | -- | -- | -- | -- | -- |
| S20 | -- | -- | -- | primary | partial | -- |
| S21 | -- | -- | -- | partial | -- | -- |
| S22 | -- | -- | primary | -- | -- | -- |
| S23 | -- | -- | -- | -- | primary | -- |
| S24 | partial | primary | -- | -- | -- | -- |
| S25 | partial | primary | -- | -- | -- | -- |
| S29b| -- | -- | -- | -- | -- | primary |
