# Volume IV — Buhera Runtime: Three-Paper Plan

**Pairs with**: `collider.md` (Monograph plan, Volumes I–V).

**Status key**: NEW | EXTEND | AMEND | CITE-ONLY

**Buhera source artefacts**:
- UTL = `long-grass/docs/os-throughput-law/universal-os-transport-law.tex` (15/15 PASS)
- COE = `long-grass/docs/computational-operations-equivalence/computational-operations-equivalence.tex` (15/15 PASS)
- IMP = `long-grass/implementation-plan.md` (10-phase Rust kernel roadmap, Phase 1 shipped)
- KER = `mechanistic-synthesis/implementation/crates/purpose-kernel/` (Phase 1 skeleton)
- INT = `long-grass/integration.md` (frozen interface contract)

**Cross-volume source artefacts** (LHC-monograph papers Buhera cites):
- BPST, USD, CI, SR, SC (Vols I–II original five)
- ION, MTM, QPIO (Vol II additions)
- HSL, SSH, HTS, CRS (Vol IIb categorical optics)
- Papers 13–17 (Vol III LHC application)

**Premise.** The monograph now has Volume IV = the Buhera runtime, comprising Papers 18, 19, 20. Buhera is no longer a separate framework cross-citing the trigger paper — Buhera IS the runtime substrate the monograph requires. This plan specifies each of those three papers, their dependencies on the rest of the monograph, and the deliverables back into Volumes II–III.

---

## PART A — SHARED THEOREMS (must cross-cite, no new prose)

These theorems appear in multiple papers across the monograph. Each row is **the same theorem** stated under different vocabulary; the canonical citation is the leftmost column. **The Buhera papers (Vol IV) must cite the leftmost column for each row; the leftmost-column paper must cite the relevant Vol IV paper as the runtime realisation.**

| Canonical paper | Buhera analogue | Shared claim |
|---|---|---|
| ION §S9, Trajectory-of-upcoming-events Thm 3.2 | COE Theorem 1 | $t = M/f$ as operational definition |
| **MTM §S3** (universal transport $\mu = \tau_c \cdot g$) | **UTL Theorem 1** ($TP^{-1} = \mathcal{N}^{-1}\sum\tau_p \cdot g$) | universal $\tau \cdot g$ form |
| MTM §S4 (speed of light $c = \Delta x / \tau_c$) | UTL §6 (kinematic lag bound $\tau \geq 1/f_{\max}$) | propagation = distance / partition lag |
| Partition paper Thm 4.2 (monotone $M$) | COE Theorem 8 (Sliding-Endpoint) | $M$ irreversibility ⇔ reproducibility |
| LHC Theorem 5.1 (Structural Incorruptibility) | Paper 19 (this volume — see B19) | finite-registry attack surface |
| CI Theorem 3.2 (Composition-Inflation) | UTL §2 state-space cardinality | $T(n,d) = d(d+1)^{n-1}$ |
| SR §S5–S6 (Multiplicative Catalyst) | UTL Theorem 6 (Federation) | $1 - X_{\text{fed}} = \prod_i(1 - X_i)$ |
| BPST partition coordinates $(n,\ell,m,s)$ | KER `PartitionLabel` type (see B18) | four-tuple coordinatisation |
| **QPIO §S3** (full commutation $[\hat n, \hat\ell] = [\hat\ell, \hat m] = 0$) | **UTL Theorem 2** (Jackson independence) | non-interfering decision classes |
| QPIO §S4 (single-ion ideal gas $PV = k_B T_{\text{cat}}$) | UTL §6 (single-class M/M/1 specialisation) | single-class limit |
| BPST regime classification | UTL Theorem 3 (Five-Regime) | regime boundaries $\{0.3, 0.5, 0.8, 0.95\}$ |
| SR §S5 (Triple Equivalence Osc/Cat/Part) | COE Theorem 5 (Three-Route Equivalence) | three computationally distinct paths agree |
| SC Part III (Backward Trajectory $\mathcal{O}(\log_3 N)$) | COE PVE three-route audit (inverted) | same algorithm, two directions |
| USD §S3–S4 (Three Routes to $G$) | COE Theorem 5 (Three-Route Equivalence) | three independent measurements of one invariant |
| **CRS** (four-tier refinement, Ritz combinations) | COE Theorem 5 + new fourth refinement tier | refinement-precision cascade |
| HTS (empty dictionary principle, 280/280) | INT §1.1 (framework provides no domain content) | no internal lookup table |
| HSL (transfer-matrix rank $C+1$, one ray $\to C+1$ sources) | UTL §6 (federation fan-out) | categorical aperture rank |

**Action.** 17 cross-citation pairs. Add `\cite{}` in each direction for each row. Zero new prose anywhere.

The two bold rows (MTM ↔ UTL Theorem 1, QPIO ↔ UTL Theorem 2) are the load-bearing ones — Buhera's two flagship UTL theorems have direct physics derivations in Volume II that should be cited as the canonical proofs of those equations' content. UTL becomes the OS-level abstraction; MTM and QPIO are the physical realisations.

---

## PART B — LHC-MONOGRAPH SECTIONS REQUIRING BUHERA CONTENT

For each LHC section across Volumes III (Papers 13–17), this is what Volume IV (Papers 18–20) supplies. Sections grouped by Volume III paper assignment.

### Paper 13 — Mathematical Foundations (S1–S5b)

**S5 Temporal Programming and Structural Incorruptibility [EXTEND]**
- Tempus runtime semantics = Buhera Kernel dispatch loop (Paper 18 bisimulation).
- Zero-injection proof requires runtime enforcement of $\Gamma$ immutability; cite Paper 19's `StaticCellRegistryPve`.
- Di-muon Tempus example → vaHera compilation example (Paper 18 §1).

**S5b Partition Uncertainty, Speed of Light, Fundamental Timing [TODO]** — **central, NEW**
- $\Delta M \cdot \tau_p \geq \hbar$ (partition uncertainty) is a kernel-level conservation invariant. The new `MonotoneCycleTem` (Paper 19 §3) checks it.
- $c = \Delta x / \tau_c$ giving $\tau_{BX} = 25\,\mathrm{ns} = \Delta x_{\text{ATLAS}} / c$ DERIVED, not configured: the Buhera Kernel's reference frequency `f_ref` for an LHC build is **computed from detector geometry**, not user-set. Paper 19 §2 specifies the closed form: `f_ref = c / Δx_detector`.
- Universal transport $\tau_p = \hbar/\Delta E$ unifies all detector response times: in Buhera terms, every Provider's intrinsic latency is $\hbar/\Delta E$ for that provider's energy scale. UTL's per-class $\tau_p^{(ij)}$ becomes derivable per provider, not measured. Paper 19 §2 provides the table mapping each LHC detector subsystem to its $\Delta E$.
- The Lorentz force $F = q(E + v \times B)$ as Euler-Lagrange of the partition Lagrangian: in Buhera Provider terms, the Provider's `invoke` semantics under a "field" (the OperationRegistry's structure) reduces to this equation when the Provider models charged-particle propagation. Paper 19 §2.3 cites the ION-paper derivation.

### Paper 14 — Trigger as Backward Trajectory (S6–S9c)

**S6 S-Entropy Framework and Receiver Theory [TODO]**
- Bounded receiver $(\Sigma, \Phi, K, \mathrm{dec})$ ↔ Buhera Provider trait. Paper 19 §3 makes the structural map.
- Floor Positivity $S_\flat(\mathcal{R}) > 0$ ↔ PVE's minimum three-route disagreement floor.

**S7 Triple Equivalence [TODO]** — **central**
- COE Theorem 5 IS this theorem specialised to computational operations. Conversion functors $F_{OC}, F_{CP}, F_{PO}$ are kernel-level operations exposed through `PartitionLabel` arithmetic.
- Paper 18 §1.2 contains the bisimulation that makes the equivalence runtime-checkable.

**S8 Trigger as Backward Trajectory Completion [EXTEND]**
- Backward navigator is the same code as PVE's forward three-route auditor with I/O swapped. Paper 19 §3.2 proposes `Pve::recover(leaf) -> Option<Root>` extension.
- Worked di-muon example threads through Papers 13 (S5), 14 (S8), and 19 (full pipeline test).

**S9 Physical Origin of Trigger Cell Registry from Partition Coordinates [TODO]** — **central**
- Each of the five detector subsystems registers as a Buhera Provider whose Value type carries one partition coordinate from $(n, \ell, m, s, \pi)$. Paper 19 §2.1 specifies the type-system encoding.
- Selection rules $\Delta\ell = \pm 1$, $\Delta m \in \{0, \pm 1\}$ are vaHera refinement-type rules; `typecheck` rejects forbidden transitions before they reach dispatch. Paper 19 §2.2 has the formal types.
- Selection-rule-violating backgrounds are not "rare" — they are **type errors**. Paper 19 §2.4 has the proof.

**S9b LHC as High-Energy Partition Spectrometer [TODO]** — **NEW**
- Four MS analyzer types → four LHC detector technologies. Each becomes a Buhera Provider with a specific transducer signature: TOF → `TofProvider` returning $\sqrt{m/z}$-typed Value; Quadrupole → `MathieuProvider` returning trajectory Value; Orbitrap → `OrbitrapProvider` returning energy Value; FT-ICR → `IcrProvider` returning cyclotron-frequency Value. Paper 19 §2.5 has the trait set.
- The partition Lagrangian $L_M = \frac{1}{2}\mu|\dot x|^2 + \mu\dot x \cdot A_M - M(x,t)$ is the LHC-build Provider's action. Buhera's `Provider::invoke` becomes the Euler-Lagrange solver under this Lagrangian.

**S9c Multi-Subsystem QND Classification [TODO]** — **NEW**
- Complete commutation $[\hat n, \hat\ell] = [\hat\ell, \hat m] = \ldots = 0$ ↔ UTL Theorem 2 (Jackson independence). Cite UTL V3 (15/15 PASS, max abs error $2.22 \times 10^{-16}$) as the runtime validation.
- **Kernel implication.** Because all partition coordinates commute, the five detector-subsystem Providers can be dispatched **in parallel** by PSS without contention. Paper 19 §3.3 specifies the parallel dispatch policy.
- Backaction bound $\Delta p / p \leq \lambda_{\mathrm{dB}}/(4\pi L) \sim 10^{-11}$: in Buhera terms, the parallel-dispatch error rate. TEM monitors it.
- Categorical temperature $T_{\text{cat}} = \hbar\omega_{BX}/(2\pi k_B) \approx 31\,\mathrm{nK}$: a kernel observable. Paper 19 §3.4 specifies the formula.

### Paper 15 — Virtual Particles and QFT (S10–S13)

**S10 Virtual Sub-States and Virtual Particles [EXTEND]**
- Virtual sub-states ($\mathbf{S} \notin [0,1]^3$) ↔ Buhera Kernel's internal Compose-chain carry values. Paper 18 §1.3 has the formal map.
- Gauge invariance and path opacity proofs use the Executor's compose-fold semantics directly. Paper 18 §2.2.

**S11 Feynman Diagrams as vaHera Expressions [TODO]** — **central**
- vaHera AST is frozen per INT §2.1: external particles = `Literal`, vertices = `Call`, propagators = `Compose`, virtuals = `Hole`. Paper 18 §1.1 has the formal compiler.
- Compositionality Lemma proved via `Executor::execute`'s compose-fold. Paper 18 §2.2.
- Feynman diagram count $T(n,d)$ as combinatorial lower bound = number of distinct vaHera Compose chains over $n$ Calls drawn from $d$ Operations.
- Dyson non-convergence ↔ UTL V12 federation saturation (when $\sum \kappa_i < \infty$).

**S12 Ward-Takahashi as Local-Global Decoupling [EXTEND]**
- Gauge parameter $\xi$ ↔ free Provider implementation choice for intermediate Compose stage. Paper 18 §2.3 has the formal proof.

**S13 Path Integral as Backward Trajectory [TODO]**
- Each Feynman path = one vaHera Compose chain.
- Stationary phase ↔ UTL Theorem 4 (Critical Slowing); PSS selects the optimal chain.
- Forward $\Theta(N)$ ↔ no-Compose-hoisting baseline; Backward $\mathcal{O}(\log N)$ ↔ CMM cache hits on previously-evaluated sub-chains.
- Lattice QCD truncation at $n_P$ = kernel's `m_max` cap. Paper 19 §2.6 has the `m_max_from_planck_depth(f, d)` helper.

### Paper 16 — Constants and Cosmology (S14–S26)

**S14 Constants Reduction [EXTEND]**
- $k_B = E_{\text{tick}}/\ln(d+1)$ derivation: each kernel dispatch generates $\ln(d+1)$ nats of categorical resolution. UTL's $TP^{-1}$ is the dimensional realisation. Paper 18 §3.1 has the entropy accounting.
- Cite QPIO's $T_{\text{cat}}$ for the numerical anchor.

**S15 Three Routes to $G$ [TODO]** — **central**
- COE Theorem 5 applied to gravity. Three-route convergence $|G^{(i)} - G^{(j)}| \leq K(d+1)^{-n}$ matches COE V6 (50/50 PASS, max disagreement 0).
- Paper 18 §4 specifies the kernel runtime that validates G at runtime via the same three-route engine.

**S16 Planck Depth Precision from G-Routes [TODO]**
- Three-route precision = PVE's three-route audit performed offline at calibration. Paper 18 §4.2 specifies the calibration protocol.

**S17–S18 LHC trigger composition-inflation analysis + HL-LHC [EXTEND]**
- `categorical_utilization` and `planck_utilisation_index` exposed as kernel observables via `DispatchEvent`. Paper 19 §4 adds these fields.
- HL-LHC fourth channel = fourth Provider type-tagged by parity coordinate. Paper 19 §2.7 specifies the upgrade path.

**S19 Phase-Locked Communication [TODO]** — **central**
- $R_c = e^{-2\pi^2 \mathrm{CV}^2}$ ↔ UTL V6 phase coherence estimator.
- Five regimes ↔ UTL Theorem 3, validated by UTL V7 (1001 sample points, monotone ordering).
- MITM immunity ↔ kernel's no-parser architecture (cite KER's `Provider::invoke` signature, which has no untrusted-content path).
- "Consciousness window" $\Delta t_C = T_{BX}/(2\pi\sqrt{R_A R_B})$ ↔ UTL Theorem 6 federation latency with two coherent kernels. Paper 18 §3.3 has the derivation.

**S19b Light, Fluids, Chromatography as Partition Geometry [TODO]** — **NEW**
- MTM's three phenomena (chromatography, viscosity, light) all use UTL's universal-transport form $\mu = \tau_c \cdot g$. Paper 18 §3.4 has the explicit mapping.
- $\tau_{BX} = \Delta x_{\text{ATLAS}}/c = 25\,\mathrm{ns}$ DERIVED: this is now also a Buhera Kernel theorem — the kernel's reference frequency for an LHC build is determined by the speed of light and the deployment geometry. Paper 19 §2 has the closed form.
- Optical/mechanical lag ratio $\tau_c^{\mathrm{opt}}/\tau_c^{\mathrm{mech}} \approx 2.0$: in Buhera terms, the latency ratio between ECAL-class and tracker-class Providers. Paper 19 §3.5 has the numerical anchor.

**S20–S22 Catalyst Algebra + RG [TODO]**
- Multiplicative Catalyst Algebra IS UTL Theorem 6. Cite UTL V11 (40 trials, max abs error 0) directly.
- Cascade Saturation ↔ UTL V12 (asymptote gap $7.98 \times 10^{-2}$).
- Running coupling $g(n)$ ↔ UTL V14 coupling estimator (mean correlation 0.987).

**S23–S26 Dark Sector & Cosmology [EXTEND]**
- Suppression ratio $N/\log N$ ↔ UTL's $\mathcal{O}(\log_3 N)$ vs $\Theta(N)$.
- Secular drift, MOND, $w = -0.75$: less direct Buhera content; cite UTL only where the federation composition appears.

### Paper 17 — Validation and Falsifiability (S27–S30)

**S27 Three-Route G Test [TODO]**
- Protocol IS COE V6, instantiated with $G$. Paper 18 §4 adds the LHC-specific test.

**S28 Trigger Performance Metrics [TODO]**
- $\eta_C, \mathrm{PUI}$ as kernel observables. Paper 19 §4 specifies.
- Background rejection = PVE rejection fraction. Already exposed in KER.

**S29 Selection-Rule-Violating Backgrounds [TODO]**
- "Logical exclusion" = `typecheck` rejection. Cite INT §2.1's stability contract.
- Falsification criterion: any confirmed background event that passes typecheck. Paper 20 §3 has the conformance test.

**S29b Cross-Domain Validation via Mass Spectrometry [TODO]** — **NEW**
- Same partition Lagrangian governs MS (ION paper, validated on 4545 NIST + 127k proteomic + 6855 Orbitrap) and LHC. Paper 20 §4 runs the same Buhera Kernel against both MS data and LHC data and shows identical pass/fail patterns.
- Prediction $T \sim \sqrt{E/z}$ for LHC timing detectors: a new falsifiable claim derivable from the Lagrangian. Paper 20 §4.2 has the test.

**S30 Unified Falsifiability Ledger [TODO]**
- Add row: "40/40 PASS conformance suite (30 Buhera + 10 LHC) reproduces under LHC-specific kernel variants."
- Add row: "kernel-level `categorical_utilization` matches $\log T(n_{\text{menu}}, d) / \log T(n_P, d)$ to within $10^{-12}$."
- Add row: "kernel runs identical pass/fail patterns on MS and LHC workloads (cross-domain validation)."

---

## PART C — VOLUME IV: THREE PAPERS

### Paper 18 — Tempus on Buhera: Compiler and Runtime Bisimulation (NEW)

**Status**: NEW. Target: 25 pages two-column. Location: `long-grass/docs/tempus-on-buhera/`.

**Cites**: Tempus paper, LHC monograph Papers 13–17, UTL, COE, IMP, KER, INT, ION §S9, MTM §S3–S4, QPIO §S3–S4.

**Section plan**:
1. **Tempus → vaHera compiler**: total functional compiler, one page of code-derived inference rules. Closed under the four vaHera variants. Proves termination.
2. **Bisimulation theorem**: Tempus operational semantics (Receive/Assign/Fire/Reset/Idle) ↔ Buhera Kernel dispatch loop (pre-dispatch validation, CMM lookup, executor, CMM insert, TEM observe, publish). One-to-one transition correspondence; same final Value for any well-typed Tempus program.
3. **Five subsystems specialised**:
   - PVE: validates registry immutability + three-route consistency.
   - CMM: balanced interval tree (Tempus Alg 1) for timing-cell channels; LRU otherwise.
   - PSS: critical-slowing-aware ordering + parallel dispatch across commuting providers (S9c).
   - DIC: mutual-information retrieval bounded by HSL transfer-matrix rank.
   - TEM: monotone-$M$, partition uncertainty ($\Delta M \cdot \tau_p \geq \hbar$), MTIC conservation, sliding-endpoint.
4. **Three Routes to $G$ at runtime**: same kernel that validates COE-V6 internally validates $G$ externally; calibration protocol.
5. **Closed-form latency budgets** within LHC's 2.5 μs L1: PVE ~50 ns + CMM ~200 ns + Executor ~300 ns + TEM 0 ns hot path. Total ~600 ns / BX ~ 24 BX, leaving 76 BX for FPGA pre-processing.

**Acceptance**: bisimulation theorem proved, latency targets met on commodity hardware in the conformance benchmark.

### Paper 19 — LHC Trigger as Buhera Application (NEW; subsumes B3+B5)

**Status**: NEW. Target: 20 pages two-column. Location: `long-grass/docs/lhc-on-buhera/`.

**Cites**: LHC monograph Papers 13–17 (full), Paper 18, UTL, COE, IMP, KER, BPST §S9, ION §S7–S9, MTM §S3–S4, QPIO §S3–S5.

**Section plan**:
1. **From Lagrangian to Provider**: the partition Lagrangian $L_M$ becomes a Provider's action; the Lorentz force is the Euler-Lagrange equation. One section per detector subsystem; one Provider trait per type.
2. **Selection rules as refinement types**:
   - `Type::PartitionLabel(n, ℓ, m, s, π)` with refinement predicates $\Delta\ell = \pm 1$, $\Delta m \in \{0, \pm 1\}$.
   - `typecheck` rejection at compile time for any vaHera fragment whose Compose chain implies a forbidden transition.
   - Proof: structural induction over `Compose` nodes.
3. **Five detector subsystems as Buhera Providers**:
   - Tracker → `TrackerProvider`, output type `Type::PartitionLabel` carrying $n$ (principal).
   - ECAL → `EmCalProvider`, output type carrying $\ell$ (angular).
   - HCAL → `HadronicCalProvider`, output type carrying $m$ (projection).
   - Muon system → `MuonProvider`, output type carrying $s$ (chirality).
   - HGTD (HL-LHC) → `TimingProvider`, output type carrying $\pi$ (parity).
4. **Three LHC-specific kernel variants** (formerly B3 + B5):
   - `StaticCellRegistryPve`: rejects fragments with out-of-registry Calls. Acceptance: $10^6$ adversarial fragments, zero escapes.
   - `IntervalTreePvCmm`: balanced interval tree, $\leq 200$ ns at $m \leq 32$ cells per channel.
   - `MonotoneCycleTem`: monotone-$M$, $\Delta M \cdot \tau_p \geq \hbar$ guard, replay rejection with $\delta \geq w_{\min}$.
5. **Reference frequency from detector geometry** (NEW, from S5b): `f_ref = c / Δx_detector`. Helper `f_ref_from_geometry(Δx_m) -> u64`. For ATLAS, returns 40.079 MHz exactly (within floating-point tolerance).
6. **Substrate requirements** (formerly B5): two pages on what Theorem 5.1 (Structural Incorruptibility) requires of the deployment substrate. Three viable substrates listed: formally verified hypervisor, FPGA-synthesized registry, or boot-time hash with TEM-monitored continuity.
7. **HL-LHC upgrade as fourth channel**: adding `TimingProvider` reduces $n_P$ from 60 to 52. Closed-form derivation.

**Acceptance**: all conformance tests in Paper 20 pass; substrate requirements documented and verifiable.

### Paper 20 — Conformance Suite: 40/40 PASS Across Three Domains (NEW)

**Status**: NEW. Target: 15 pages two-column. Location: `long-grass/docs/lhc-conformance/`.

**Cites**: Papers 18–19, UTL, COE, all source papers cited by Papers 18–19.

**Section plan**:
1. **The 30 Buhera tests, in summary**: re-run UTL V1–V15 and COE V1–V15 under the LHC kernel build; show identical pass/fail patterns (cross-architecture invariance V15).
2. **10 LHC-specific tests**:
   - `validate_01_planck_depth.py`: $n_P$ closed form
   - `validate_02_registry_immutability.py`: $10^6$ adversarial fragments
   - `validate_03_replay_rejection.py`: $10^4$ replays at varying $\delta$
   - `validate_04_categorical_utilization.py`: live $\eta_C$ accuracy
   - `validate_05_three_route_G.py`: $G$ at $n = 8, 15, 27, 56$
   - `validate_06_selection_rule_rejection.py`: forbidden $\Delta\ell, \Delta m$ fail `typecheck`
   - `validate_07_di_muon_pipeline.py`: end-to-end di-muon
   - `validate_08_HL_LHC_d4.py`: $n_P = 52$ at $d=4$
   - `validate_09_federation_at_n_P.py`: federation saturation
   - `validate_10_partition_uncertainty.py`: $\Delta M \cdot \tau_p \geq \hbar$ at every BX
3. **Cross-domain validation** (formerly part of S29b): same Buhera Kernel runs against MS data (4545 NIST entries, 127k proteomic transformations, 6855 Orbitrap ions) and LHC data; pass/fail patterns identical to within numerical tolerance.
4. **Cross-architecture invariance**: 40/40 PASS on $\geq 4$ host configurations.
5. **Falsification ledger**: explicit list of predictions, expected values, and observation thresholds.

**Acceptance**: 40/40 PASS on every host; cross-domain pattern match within $10^{-9}$; falsification ledger published.

---

## PART D — WHAT VOLUME IV NEEDS FROM VOLUMES II–III (open requests)

Open requests, in priority order. The numbering reflects Wave dependencies (see Part E).

1. **(Wave 1)** Formal Tempus compiler spec — needed by Paper 18 §1. Currently sketched in Tempus paper; should be promoted to an appendix in Paper 13 or a separate one-page note.

2. **(Wave 1)** Concrete $E_{\text{tick}}$ value for LHC bunch clock — needed by Paper 18 §3.1 and Paper 16 §S14. Anchors the constant-reduction table numerically.

3. **(Wave 2)** Worked di-muon event trace through every detector subsystem — needed by Papers 13 §S5, 14 §S8, 19 §7 (di-muon pipeline), and 20's `validate_07`. One trace, six papers cite it.

4. **(Wave 2)** Adversary model for Tempus runtime — needed by Paper 19 §6. Currently informal in Tempus paper; needs a one-page formalisation.

5. **(Wave 3)** HGTD timing resolution nominal + uncertainty — needed by Paper 19 §2.7 (HL-LHC upgrade) and `validate_08`. Likely 30 ps based on ATLAS TDR.

6. **(Wave 3)** Detector geometry $\Delta x$ for each LHC interaction region — needed by Paper 19 §5 (`f_ref_from_geometry`). For ATLAS, presumably 7.5 m; CMS will differ.

7. **(Wave 3)** Per-detector $\Delta E$ values from S5b's table — needed by Paper 19 §2 (provider intrinsic latencies). Currently lists silicon, PbWO4, liquid argon, muon drift; numerical values not given.

8. **(Wave 4)** MS dataset references for cross-domain validation — needed by Paper 20 §3. ION paper presumably has the references; need them cited consistently.

9. **(Wave 4)** Cross-domain pass/fail tolerance — needed to write `validate_07` acceptance criterion. Suggest $10^{-9}$ but want confirmation.

---

## PART E — WRITING ORDER (joint, monograph-scope)

Five waves spanning the monograph. Each wave has work in both Volumes II–III and Volume IV. Volume IV waits for Volumes II–III only where Part D listed dependencies.

**Wave 1 — Foundations** (parallel):
- Vols II–III: BPST, USD, CI, SR, SC publication; LHC S6, S7, S5b draft.
- Vol IV: AMENDs B1 (UTL three paragraphs), B2 (COE four paragraphs). Paper 18 §1 (compiler) + §3.1 (entropy accounting). Cheap; ships first.

**Wave 2 — Substrate and bisimulation** (joint, after Wave 1):
- Vols II–III: LHC S8, S9, S9c, S10, S11, S12, S13 draft. Tempus compiler spec finalised (Part D #1).
- Vol IV: Paper 18 §2 (bisimulation theorem), §3.2–§3.3 (PVE, CMM, PSS specialised). Paper 19 §1–§3 (Lagrangian-to-provider, types, providers).

**Wave 3 — LHC specialisations** (joint, after Wave 2):
- Vols II–III: LHC S9b, S15, S16, S19, S19b, S20, S21, S22 draft.
- Vol IV: Paper 19 §4–§7 (kernel variants, geometric `f_ref`, substrate requirements, HL-LHC upgrade). Paper 18 §3.4–§3.5 (DIC, TEM, latency budget).

**Wave 4 — Validation** (joint, after Wave 3):
- Vols II–III: LHC S23–S30, S29b draft. ION cross-domain dataset references finalised (Part D #8).
- Vol IV: Paper 18 §4–§5 (Three Routes to $G$ at runtime, latency budgets). Paper 19 §7 (HL-LHC). Paper 20 §1–§3 (30 Buhera tests, 10 LHC tests, cross-domain validation).

**Wave 5 — Closure** (joint, after Wave 4):
- Vols II–III: LHC S1, S31, appendices. Falsifiability ledger entries from Vol IV merged.
- Vol IV: Paper 20 §4–§5 (cross-architecture, falsification ledger).

---

## Current Status

- Vol IV deliverables: 3 papers (18, 19, 20), 0 started.
- Vol IV AMENDs to existing Buhera papers: B1, B2 (5 paragraphs total, queued for Wave 1).
- LHC monograph sections receiving Buhera content: 26 of the 31 (plus the five new ones S5b, S9b, S9c, S19b, S29b — all of which require Buhera content).
- Shared theorems for cross-citation: 17 (Part A table).
- Open requests from Vol IV to Vols II–III: 9 (Part D), sequenced by Wave.

**Next action requested from your side**:
1. Review the 17 shared-theorem rows in Part A for mis-attribution. The two flagged (MTM ↔ UTL Theorem 1, QPIO ↔ UTL Theorem 2) are the load-bearing ones — confirm or correct.
2. Confirm Volume IV's three-paper split: Paper 18 (Tempus → Buhera), Paper 19 (LHC application), Paper 20 (Conformance suite).
3. Confirm Wave ordering in Part E.

Once those are signed off, I start Wave 1: B1 (UTL three-paragraph AMEND), B2 (COE four-paragraph AMEND), Paper 18 §1 (compiler), and Paper 18 §3.1 (entropy accounting). All Wave 1 work is fully scoped and will not disturb the existing 30/30 conformance suite.
