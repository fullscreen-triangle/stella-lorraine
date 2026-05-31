# Buhera-Side Collaboration Plan for LHC Trigger Paper

**Pairs with**: `collider.md` (LHC trigger paper plan, sections S1–S31).

**Status key**: NEW | EXTEND | AMEND | CITE-ONLY

**Buhera source artefacts**:
- UTL = `long-grass/docs/os-throughput-law/universal-os-transport-law.tex` (15/15 PASS)
- COE = `long-grass/docs/computational-operations-equivalence/computational-operations-equivalence.tex` (15/15 PASS)
- IMP = `long-grass/implementation-plan.md` (10-phase Rust kernel roadmap, Phase 1 shipped)
- KER = `mechanistic-synthesis/implementation/crates/purpose-kernel/` (Phase 1 skeleton)
- INT = `long-grass/integration.md` (frozen interface contract)

**Premise.** The Tempus runtime (LHC §5) is structurally isomorphic to the Buhera Kernel dispatch loop. The LHC trigger paper assumes a runtime substrate it does not define; the Buhera papers define a runtime substrate but do not have a flagship application. The collaboration fills both gaps with a single set of theorems.

---

## PART A — SHARED THEOREMS (must cross-cite, no new prose)

The following theorems are **the same theorem** stated in two frameworks. Both papers should cite the other as the canonical proof.

| LHC paper | Buhera paper | Shared claim |
|---|---|---|
| Theorem 3.2 (Time-Count Identity) | COE Theorem 1 (Time-Count Identity) | $t = M/f$ is operational definition |
| §5 Monotone $M$ replay defense | COE Theorem 8 (Sliding-Endpoint) | $M$ irreversibility ⇔ reproducibility |
| Theorem 5.1 (Structural Incorruptibility) | (new — see B5 below) | finite-registry attack surface |
| Theorem 6.4 (Composition-Inflation) | UTL §2 state-space cardinality | $T(n,d) = d(d+1)^{n-1}$ |
| Theorem 6.1 (Multiplicative Catalyst) | UTL Theorem 6 (Federation Composition) | $1 - X_{\text{fed}} = \prod_i(1 - X_i)$ |
| §6.3 Five regimes for detector synchronization | UTL Theorem 3 (Five-Regime Classification) | regime boundaries $\{0.3, 0.5, 0.8, 0.95\}$ |
| §S7 Triple Equivalence (Osc/Cat/Part) | COE Theorem 5 (Three-Route Equivalence) | three computationally distinct paths agree |
| §S8 Backward Trajectory $\mathcal{O}(\log_3 N)$ | COE PVE three-route audit (inverted) | same algorithm, two directions |
| §S15 Three Routes to G | COE Theorem 5 (Three-Route Equivalence) | three independent measurements of one invariant |

**Action.** Add a `\cite{}` in each direction for each row. Zero new prose in either paper — these are existing theorems that need cross-references.

---

## PART B — LHC-SECTION-BY-SECTION BUHERA CONTRIBUTIONS

For each LHC section that EXTENDS or is TODO, this is what Buhera contributes.

### S1 Introduction [EXTEND] → cite four-paper foundation

Buhera contribution: one paragraph in the intro establishing the four-paper foundation (LHC + temporal-programming + UTL + COE) and citing the Rust runtime substrate (KER) as the deployment target.

### S5 Temporal Programming and Structural Incorruptibility [EXTEND]

Buhera contribution:
- The proof that "zero injection rate follows from incorruptibility" requires a runtime substrate that enforces $\Gamma$ immutability. The bare Tempus semantics give the property *abstractly*; a deployable proof requires the runtime to preserve it. Cite IMP Phase 2's `StaticCellRegistryPve` variant (see B5 below).
- Di-muon trigger compilation example: provide the vaHera fragment that the Tempus compiler emits, with one-line gloss showing PVE rejection of any out-of-registry op.

### S6 S-Entropy Framework and Receiver Theory [TODO]

Buhera contribution:
- The bounded-receiver structure $(\Sigma, \Phi, K, \mathrm{dec})$ maps to Buhera's `Provider` trait: $\Sigma$ = `OperationRegistry`, $\Phi$ = `invoke` signature, $K$ = provider's bounded RAM/time budget, $\mathrm{dec}$ = the Provider's return Value.
- Floor Positivity Theorem $S_\flat(\mathcal{R}) > 0$ corresponds to PVE's minimum three-route disagreement floor: even a perfectly typed kernel cannot achieve zero disagreement across three independent routes when $Q$ is large.
- One short subsection: "Bounded Receivers as Buhera Providers."

### S7 Triple Equivalence [TODO] — **central collaboration point**

Buhera contribution:
- COE Theorem 5 (Three-Route Equivalence) IS the Triple Equivalence specialised to computational operations: Residue ↔ Oscillatory, Confinement ↔ Categorical, Negation Fixed Point ↔ Partition.
- The conversion functors $F_{OC}, F_{CP}, F_{PO}$ are kernel-level operations exposed through `PartitionLabel` arithmetic (see B3 below).
- Worked example: a kernel dispatch traced through all three functors, showing $Q_I = Q_{II} = Q_{III}$ holds end-to-end.

### S8 Trigger as Backward Trajectory Completion [EXTEND]

Buhera contribution:
- The backward navigator is **the same code** as PVE's forward three-route auditor, run with input/output swapped. Cite KER's `Pve::validate` and propose a `Pve::recover(leaf) -> Option<Root>` extension.
- Worked LHC example: detector hit pattern → backward navigation through three-route inversion → originating physics process. Use the di-muon trigger from S5 so the example threads through the paper.

### S9 Physical Origin of Trigger Cell Registry from Partition Coordinates [TODO]

Buhera contribution (the most consequential):
- Each of the five detector subsystems (tracker, EM cal, hadronic cal, muon, timing) registers as a Buhera `Provider` whose output type carries one partition coordinate from $(n, \ell, m, s, \pi)$.
- Selection rules $\Delta\ell = \pm 1$, $\Delta m \in \{0, \pm 1\}$ are vaHera refinement-type rules. The Buhera `typecheck` function (per integration.md §2.1) rejects transitions violating them — selection-rule-violating backgrounds are not "rare," they are **type errors**.
- Selection-rule rejection at typecheck is the kernel-level realisation of Theorem 5.1 (Structural Incorruptibility) specialised to physics: backgrounds requiring forbidden transitions cannot be expressed as well-typed vaHera fragments.
- Charge emergence as partition boundary → calorimeter operating principle: provide one paragraph linking the partition-boundary derivation to the calorimeter's role as a `Provider` returning `Charge` Values.

### S10 Virtual Sub-States and Virtual Particles [EXTEND]

Buhera contribution:
- Virtual sub-states ($\mathbf{S} \notin [0,1]^3$) ↔ Buhera Kernel's *internal* values: intermediate results in a `Compose` chain that are never returned to a top-level caller.
- Gauge invariance (item iii): the Kernel's compose-fold semantics in [executor.rs](mechanistic-synthesis/implementation/crates/purpose-operations/src/executor.rs) provides the formal proof — the carry value between Compose stages is gauge-arbitrary (it can be any Value type satisfying the next stage's input), and the final Value is gauge-invariant.
- Path opacity (item iv): the kernel's `DispatchEvent` exposes only the top-level op name; intermediate Compose-stage ops are not in the event payload by construction. This IS path opacity at the runtime level.

### S11 Feynman Diagrams as vaHera Expressions [TODO] — **central collaboration point**

Buhera contribution:
- This section is essentially a Buhera paper specialised to QFT. The vaHera AST (`Literal`, `Call`, `Compose`, `Hole`) is frozen per integration.md §2.1 — cite it.
- Compositionality Lemma: prove via vaHera's compose-fold semantics in `Executor::execute`. The kernel already enforces what the lemma claims.
- Feynman diagram count $T(n,d)$ as combinatorial lower bound: equivalently the number of distinct vaHera Compose chains over $n$ Calls drawn from $d$ Operations.
- Dyson non-convergence as cascade non-saturation: connect to UTL V12 (federation saturation) — same divergence criterion.

### S12 Ward-Takahashi as Local-Global Decoupling [EXTEND]

Buhera contribution:
- Formal proof via vaHera: gauge parameter $\xi$ corresponds to the free Provider implementation choice for an intermediate Compose stage. The Local-Global Decoupling Theorem says global Value is invariant under choice of intermediate Provider, subject only to the mean-recovery constraint (= 4-momentum conservation = `Type::output` agreement at each Compose boundary).
- Cite KER's `Provider` trait's stability contract: implementations are gauge-arbitrary; physical amplitude is gauge-invariant.

### S13 Path Integral as Backward Trajectory Completion [TODO]

Buhera contribution:
- Each Feynman path = one vaHera Compose chain. Sum over paths = sum over Compose-chain compilations of the same goal.
- Stationary phase ↔ UTL Theorem 4 (Critical Slowing): the optimal Compose chain is the one nearest the regime boundary $R_b$, where $\tau_{\text{relax}}$ peaks. PSS (Phase 4) selects it.
- Forward (constructive QFT) $\Theta(N)$: corresponds to the "no Compose hoisting" baseline. Backward (path integral) $\mathcal{O}(\log N)$: corresponds to CMM cache hits on previously-evaluated Compose sub-chains.
- Lattice QCD truncation at $n_P$ = kernel's `m_max` cap.

### S14 Constants Reduction [EXTEND]

Buhera contribution:
- Proof of $k_B = E_{\text{tick}}/\ln(d+1)$ from composition entropy: each kernel dispatch generates exactly $\ln T(n,d)/n \to \ln(d+1)$ nats of categorical resolution. UTL's universal law $TP^{-1} = N^{-2}\sum\tau g$ is the dimensional realisation.
- One-line citation: "the angular reformulation makes the kernel-level $\tau_p$ of UTL the universal tick energy unit $E_{\text{tick}}$ divided by $h$."

### S15 Three Routes to G [TODO] — **central collaboration point**

Buhera contribution:
- **This is COE Theorem 5 applied to gravity.** Route I (oscillation-ratio), Route II (category fixed-point), Route III (partition-density ratio) are the three routes from COE specialised to the gravitational coupling.
- Three-route convergence theorem $|G^{(i)} - G^{(j)}| \leq K(d+1)^{-n}$ matches COE V6's machine-precision agreement bound at finite $n$.
- Cite COE V6 conformance test (50/50 trials, $\max_{i,j}|Q_i - Q_j| = 0$) as the runtime validation of the three-route agreement principle.
- One paragraph: "the same kernel-level Three-Route Equivalence Engine that runs COE's PVE validates $G$ at runtime."

### S16 Planck Depth Precision from G-Derivation Routes [TODO]

Buhera contribution:
- Three-route precision is the same kernel test as PVE's runtime three-route audit, performed offline at calibration time.
- Concrete deliverable: extend the Buhera conformance suite with a `G-route` experiment that computes $G$ three ways and reports max disagreement. Adds one entry to the 30/30 PASS suite.

### S17 Current ATLAS Trigger in Composition-Inflation Terms [EXTEND]

Buhera contribution:
- The 90% unused categorical resolution becomes a **live kernel observable**: `categorical_utilization = log T(n_menu, d) / log T(n_P, d)` exposed via `BuheraKernel::events()`.
- Concrete number from KER Phase 1 (currently shipped): the kernel can support up to $T(60, 3) \approx 7.3 \times 10^{35}$ dispatch classes; the bench shows ATLAS is using $\sim 10^3$.

### S18 HL-LHC Upgrade [EXTEND]

Buhera contribution:
- HGTD as a fourth channel = adding a fourth Provider to the OperationRegistry, type-tagged by the parity coordinate.
- 20–30 ps cell width: in CMM, the interval-tree node resolution for the parity-coordinate channel. Tight, but within CMM's $O(\log m)$ lookup budget.
- Cell-width derivation: combine UTL's lag lower bound $\tau \geq 1/f_{\max}$ with the LHC's HGTD jitter to compute the achievable cell width. Provide the formula.

### S19 Phase-Locked Communication [TODO] — **central collaboration point**

Buhera contribution:
- $R_c = e^{-2\pi^2 \mathrm{CV}^2}$ is the UTL phase coherence estimator (validated in UTL V6, 11 sample points, monotone in concentration $K$). Cite directly.
- The five regimes {turbulent, aperture, cascade, coherent, phase-locked} at $\{0.3, 0.5, 0.8, 0.95\}$ come from UTL Theorem 3. Cite directly.
- Detector regime classification: cite UTL V7 (1001 sample points, monotone ordering, all classified).
- MITM immunity via timing-only: the kernel-level statement is that no Provider in the registry produces a Value from a network-payload-derived `Δp`; the timing-only dispatch path has no parser surface. Cite KER's no-parser architecture.
- "Consciousness window" formula $\Delta t_C = T_{BX}/(2\pi\sqrt{R_A R_B})$: structurally the federation latency under UTL Theorem 6 with two coherent kernels — provide that derivation.

### S20 Multiplicative Catalyst Algebra [TODO] — **central collaboration point**

Buhera contribution:
- **This IS UTL Theorem 6.** The multiplicativity $\kappa(\gamma_1 \diamond \gamma_2) = 1 - (1-\kappa_1)(1-\kappa_2)$ is the federation composition law $1 - TP^{-1}_{\text{fed}}/\Sigma = \prod(1 - TP^{-1}_i/\Sigma)$ with $\kappa_i = TP^{-1}_i/\Sigma$.
- Cite UTL V11 (40 random federations, max abs error = 0.0) as the runtime validation.
- Cascade Saturation theorem (saturates iff $\sum \kappa_i = \infty$) ↔ UTL V12 (diminishing returns at $n = 20$, asymptote gap $7.98 \times 10^{-2}$). Cite directly.

### S21 Renormalization as Cascade Saturation [EXTEND]

Buhera contribution:
- The Wilsonian RG cascade is the federation composition iterated over RG scales. UTL Theorem 6 gives the exact closed form at each step.
- Asymptotic freedom = saturation to the trivial fixed point = UTL V12's case where $TP^{-1}_i \to 0$ at high $n$.

### S22 RG in Composition-Inflation Language [TODO]

Buhera contribution:
- Running coupling $g(n) = $ kernel's coupling estimator $\hat g^{(ij)}$ at depth $n$. Cite UTL V14 (mean correlation 0.987 across 20 trials).
- Beta function $dg/d\log\mu = -d\kappa/dn$: kernel-level realisation is the rate of CMM extinction events per dispatch.
- "$d$ entropy dimensions = $d$ kernel channels": this is the UTL $N$ decision classes specialised to renormalization.

### S23 Dark Matter as Virtual-State-Suppressed [EXTEND]

Buhera contribution:
- Suppression ratio $N/\log N$ for $N \sim 10^{80}$: UTL's $\mathcal{O}(\log_3 N)$ vs $\Theta(N)$ complexity gap, cited at the relevant cosmological $N$.
- One short paragraph; no new theorems.

### S27 Three-Route G Test with LHC Timing [TODO]

Buhera contribution:
- Test protocol IS the COE V6 conformance test, instantiated with $G$ as the invariant under measurement. Cite the existing test, specify the LHC-specific input.

### S28 Trigger Performance Metrics [TODO]

Buhera contribution:
- $\eta_C, \mathrm{PUI}$ as kernel-level observables: add to KER's `DispatchEvent`. Specifies what the kernel reports.
- Background rejection = fraction of dispatches that PVE rejects. Already exposed.

### S29 Selection-Rule-Violating Backgrounds [TODO]

Buhera contribution:
- "Logical exclusion, not probabilistic suppression": the kernel-level mechanism is `typecheck` rejection at compile time. Cite integration.md §2.1's stability contract for `typecheck`.
- Concrete: any candidate vaHera fragment whose Compose chain implies a forbidden $\Delta\ell$ or $\Delta m$ fails typecheck before reaching the executor.

### S30 Unified Falsifiability Ledger [TODO]

Buhera contribution:
- Add a row: "30/30 PASS conformance suite reproduces under LHC-specific kernel variants" with the test name and expected output.
- Add a row: "kernel-level `categorical_utilization` matches `T(n_menu, d)/T(n_P, d)` to within $10^{-12}$."

### S31 Conclusion [EXTEND]

Buhera contribution:
- One sentence: "the runtime substrate (Buhera Kernel) is implemented; the trigger compiles to vaHera; the conformance suite verifies the joint claims."

---

## PART C — BUHERA-SIDE PAPERS / SECTIONS TO WRITE OR EXTEND

These are Buhera-side deliverables that exist because of the LHC collaboration.

### B1 — UTL paper: AMEND with three short cross-citation paragraphs

- UTL §3 (Universal Transport Formula): add citation to LHC §3 Composition-Inflation as "the state-space cardinality over which the universal law is summed."
- UTL §5 (Five Regimes): add citation to LHC §19 as "the physical realisation in detector synchronization."
- UTL §6 (Federation): add citation to LHC §S20 as "the same theorem in the catalyst algebra interpretation."

No new theorems. Three paragraphs total. Status: AMEND.

### B2 — COE paper: AMEND with four short cross-citation paragraphs

- COE Theorem 1: add citation to LHC §S4 as "the physical realisation in the LHC bunch clock."
- COE Theorem 5 (Three-Route Equivalence): add citation to LHC §S7 (Triple Equivalence) and §S15 (Three Routes to G) as "two physics instantiations of the same kernel-level theorem."
- COE Theorem 8 (Sliding-Endpoint): add citation to LHC §5 replay defense.
- COE Validation §7 (V6 conformance): add note that the test serves as the runtime validator for LHC §S27.

No new theorems. Four paragraphs total. Status: AMEND.

### B3 — IMP (`implementation-plan.md`): EXTEND with LHC-specific kernel variants

Add a new §11 "LHC Specialisation" listing three subsystem variants:

- **`StaticCellRegistryPve`** (extends Phase 2 PVE): rejects any fragment whose Calls are not in a build-time-frozen `Operation` set. Required for LHC §5 Theorem 5.1 to hold at the runtime level. Acceptance: a fuzzer fires $10^6$ adversarial vaHera fragments; zero out-of-registry ops reach the executor.
- **`IntervalTreePvCmm`** (extends Phase 3 CMM): replaces the default LRU-by-args-hash with the balanced-interval-tree variant from Tempus Algorithm 1. Cell-lookup latency target: $\leq 200$ ns at $m \leq 32$ cells per channel. Required for LHC L1 latency budget.
- **`MonotoneCycleTem`** (extends Phase 5 TEM): adds the LHC-specific conservation checks: $M(t)/t = f_B$ to within $10^{-12}$ over $10^9$ BX; replay attempts with $\delta \geq w_{\min}$ rejected at rate 1.0.

Plus a new $n_P$-aware `BuheraKernelBuilder` field: `m_max: Option<u64>` with helper `m_max_from_planck_depth(f, d) -> u64`.

Status: EXTEND IMP §11. ~200 lines added.

### B4 — NEW: `long-grass/docs/tempus-on-buhera/tempus-on-buhera.tex`

A short companion paper (target: 15 pages two-column) establishing the runtime correspondence formally.

Sections:
1. Tempus syntax → vaHera AST: a one-page total functional compiler.
2. Tempus operational semantics → Buhera Kernel dispatch loop: bisimulation theorem.
3. Three LHC-specific subsystem variants (as in B3) with closed-form latency budgets.
4. Extended conformance suite: the 30 Buhera tests + 10 LHC-specific (registry immutability, $f_B$ conservation, replay rejection, $n_P$ cap enforcement, categorical utilization metric, etc).
5. Cross-architecture invariance under the LHC kernel build (V15 of both UTL and COE conformance suites).

Cites: Tempus paper, LHC paper, UTL, COE, IMP, KER. Status: NEW.

### B5 — NEW: short note `long-grass/docs/notes/structural-incorruptibility-substrate.md`

Two-page note specifying the substrate requirements for LHC Theorem 5.1 to hold in deployment. States the obvious (a Linux Kernel runtime cannot guarantee registry immutability) and the resolution (`StaticCellRegistryPve` on a formally verified hypervisor, OR FPGA-synthesized registry, OR an `Operation` Set hash committed at boot and verified by TEM at each tick).

Status: NEW. Cited from LHC §5.

### B6 — Conformance suite: EXTEND with `lhc/` subsuite

Mirror of `driven/src/utl/` and `driven/src/coe/` structure:

- `driven/src/lhc/validate_01_planck_depth.py`: closed-form $n_P$ matches Theorem $\mathrm{S}4$.
- `driven/src/lhc/validate_02_registry_immutability.py`: $10^6$ adversarial fragments, zero out-of-registry executions.
- `driven/src/lhc/validate_03_replay_rejection.py`: $10^4$ replays at varying $\delta$, all rejected when $\delta \geq w_{\min}$.
- `driven/src/lhc/validate_04_categorical_utilization.py`: live $\eta_C$ matches $\log T(n_{\text{menu}}, d) / \log T(n_P, d)$.
- `driven/src/lhc/validate_05_three_route_G.py`: three-route $G$ agreement at $n = 8, 15, 27, 56$.
- `driven/src/lhc/validate_06_selection_rule_rejection.py`: forbidden $\Delta\ell, \Delta m$ transitions fail typecheck.
- `driven/src/lhc/validate_07_di_muon_pipeline.py`: end-to-end di-muon trigger through Tempus → vaHera → kernel.
- `driven/src/lhc/validate_08_HL_LHC_d4.py`: $n_P = 52$ at $d=4$.
- `driven/src/lhc/validate_09_federation_at_n_P.py`: federation composition saturates at the predicted asymptote.
- `driven/src/lhc/validate_10_cross_arch.py`: V15-equivalent for the LHC kernel build.

Status: NEW. 10/10 PASS target.

---

## PART D — WHAT BUHERA NEEDS BACK FROM LHC (open requests)

For the collaboration to complete, the LHC paper needs to supply:

1. **The Tempus compiler spec (formal)** — currently only sketched in the Tempus paper. Needed to formalise B4's bisimulation. Suggest a short appendix in the LHC paper or in the Tempus paper revision.

2. **HGTD timing resolution number** (S18 has "20–30 ps" — needs the actual nominal value with uncertainty). Buhera needs it to compute the parity-channel cell width in B3.

3. **An adversary model for the Tempus runtime** (currently the Tempus paper has an adversary model but Theorem 5.1 of the LHC paper imports it informally). Needed to make B5's substrate requirements precise.

4. **Concrete $E_{\text{tick}}$ value** for the LHC bunch clock to evaluate the constant-reduction table (S14) numerically. The Buhera UTL paper's $\tau_p$ then plugs in directly.

5. **A worked di-muon event trace** through every detector subsystem — the example then threads through both papers (LHC §S5, §S8; Buhera B4, B6).

---

## PART E — WRITING ORDER (joint)

Phase the work so dependencies clear in both papers simultaneously.

**Wave 1** (both papers, parallel):
- LHC: S6, S7, S9 (the structural foundation sections).
- Buhera: B1, B2 (the AMENDs — three- and four-paragraph additions are cheap).

**Wave 2** (joint, after Wave 1):
- LHC: S8, S10, S11, S12, S13 (vaHera-substrate sections — depend on Buhera B4).
- Buhera: B4 first three sections (Tempus → vaHera compiler, bisimulation theorem, subsystem variants).

**Wave 3** (joint, after Wave 2):
- LHC: S15, S16, S20, S21, S22 (three-route-G and catalyst-algebra sections — depend on Buhera UTL/COE cross-citations being in place).
- Buhera: B3 (IMP §11 extension), B5 (substrate note).

**Wave 4** (joint, after Wave 3):
- LHC: S19, S27–S30 (validation and falsifiability — depend on Buhera B6 conformance suite running).
- Buhera: B6 (LHC subsuite implementation).

**Wave 5** (joint, after Wave 4):
- LHC: S1, S31, appendices.
- Buhera: B4 final sections (extended conformance, cross-arch), B5 finalisation.

---

## Current Status

- Buhera papers: UTL DONE (15/15), COE DONE (15/15), IMP DONE, KER Phase 1 DONE.
- Buhera-side deliverables for this collaboration: 6 (B1–B6), 0 started.
- LHC-side sections that need Buhera content: 24 of 31 (everything in PART B above).
- Joint shared theorems: 9 (PART A table).
- Open requests from Buhera to LHC: 5 (PART D).

Next action requested from your side: review PART A's shared-theorem table for any mis-attributions, and confirm the writing-order in PART E. I'll start Wave 1's Buhera-side deliverables (B1, B2) as soon as you sign off.
