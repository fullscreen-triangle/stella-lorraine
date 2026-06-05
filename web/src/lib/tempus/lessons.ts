// ── Tempus tutorial curriculum ────────────────────────────────────────────────
// A language-only arc of increasing complexity. Each lesson carries explanatory
// prose, a runnable/compilable script, and simulation hints. The scripts parse
// under the surface grammar in lexer.ts / parser.ts (cell / sync / compose /
// when, with emit / fire / begin…end statements).

export interface Lesson {
  id:      string;
  title:   string;
  tagline: string;
  body:    string[];        // concept paragraphs
  points:  string[];        // key takeaways / things to try
  script:  string;
  events:  number;          // default RUN length
  noise:   number;          // default RUN noise σ (0..1)
  expect:  'clean' | 'diagnostics';
  feature?: 'inject' | 'phase';   // optional lesson-specific panel
  kind?:   'example';       // instrument example (vs. numbered lesson)
}

export const LESSONS: Lesson[] = [
  // ── 01 ──────────────────────────────────────────────────────────────────────
  {
    id: 'hello-dp',
    title: 'Hello, ΔP',
    tagline: 'The timing deviation is the only datum.',
    body: [
      'In an ordinary protocol a message carries content — bytes that must be parsed. In Tempus there is no content. The single runtime datum is the timing deviation of a pulse from the reference oscillator’s predicted tick:',
      'ΔP(k) = T_ref(k) − t_rec(k).',
      'A positive ΔP means the k-th pulse arrived early; negative means late; zero means perfect synchrony. A Tempus program is a set of cells — intervals in ΔP-space — each mapping to a pre-compiled action. Execution is: measure ΔP, find its cell, dispatch that cell’s action.',
      'Below, a 10 MHz oscillator feeds one channel. Two cells split ΔP into a SIGNAL band near zero and a NOISE band further out. Press COMPILE to freeze the cell registry Γ, then RUN to watch pulses get classified.',
    ],
    points: [
      'COMPILE checks the program and freezes the registry Γ.',
      'RUN simulates a stream of pulses and dispatches the matching action.',
      'Nothing about a pulse matters except when it arrived.',
    ],
    script: `-- 01 · Hello, ΔP
-- The only datum is ΔP(k) = T_ref(k) − t_rec(k):
-- how far the k-th pulse landed from the oscillator's tick.

sync clock at 10.0e6 freq            -- 10 MHz reference oscillator

cell SIGNAL bounds (-1.0e-7, 1.0e-7) action 0   -- |ΔP| < 100 ns
cell NOISE  bounds ( 1.0e-7, 1.0e-5) action 1   -- off to one side

compose d=1 channels clock into stream

when SIGNAL do emit accept
when NOISE  do emit reject`,
    events: 400,
    noise: 0.4,
    expect: 'clean',
  },

  // ── 02 ──────────────────────────────────────────────────────────────────────
  {
    id: 'partition',
    title: 'Tiling ΔP-space',
    tagline: 'A program is a partition; coverage is a compile-time property.',
    body: [
      'A cell is a measurable interval with positive width — never a single point, because oscillator jitter makes point predicates undetectable. A well-formed program tiles the relevant range of ΔP-space: the cells cover it with no gaps and no overlaps.',
      'Here five cells tile a symmetric window around zero, reading ΔP as a phase error: LOCKED in the centre, two DRIFT bands flanking it, and LATE / EARLY at the extremes. The sign of ΔP now carries meaning — early vs. late — and each band dispatches a corrective action.',
      'COMPILE reports the coverage span and confirms the tiling is complete. Try widening LOCKED, or deleting DRIFT_HI, and re-compile to see how the coverage report changes.',
    ],
    points: [
      'Cells must have positive width and should tile without gaps.',
      'The sign of ΔP distinguishes early from late.',
      'COMPILE reports the covered span [min lo, max hi].',
    ],
    script: `-- 02 · Tiling ΔP-space
-- Five contiguous cells partition a phase-error window.
-- ΔP < 0 late, ΔP > 0 early.

sync clock at 10.0e6 freq

cell LATE     bounds (-1.0e-5, -5.0e-7) action 2
cell DRIFT_LO bounds (-5.0e-7, -1.0e-7) action 1
cell LOCKED   bounds (-1.0e-7,  1.0e-7) action 0
cell DRIFT_HI bounds ( 1.0e-7,  5.0e-7) action 1
cell EARLY    bounds ( 5.0e-7,  1.0e-5) action 2

compose d=1 channels clock into stream

when LOCKED   do emit locked
when DRIFT_LO do emit nudge_back
when DRIFT_HI do emit nudge_fwd
when LATE     do fire resync
when EARLY    do fire resync`,
    events: 600,
    noise: 0.5,
    expect: 'clean',
  },

  // ── 03 ──────────────────────────────────────────────────────────────────────
  {
    id: 'diagnostics',
    title: 'Reading diagnostics',
    tagline: 'COMPILE rejects ill-formed programs before they ever run.',
    body: [
      'The compiler is the safety boundary. Before any event is processed it checks the static semantics: every `when` must name a declared cell, every composed channel must have a sync source, and the cell partition is examined for gaps and overlaps. A program that fails these checks emits diagnostics and does not freeze a registry.',
      'This program has three planted problems. Press COMPILE and read the panel: there is a coverage gap between two cells, a `when` that names a cell which does not exist (a typo), and a cell with no firing rule.',
      'Fix them — close the gap by adjusting a bound, correct the misspelled cell name — and COMPILE again until the panel is clean. This is the loop you will use for every program.',
    ],
    points: [
      'Errors block compilation; warnings and info do not.',
      'Undeclared-reference errors suggest the nearest declared name.',
      'A coverage gap means some ΔP values fall through to the anomaly action.',
    ],
    script: `-- 03 · Reading diagnostics  (this program has bugs — fix them)

sync clock at 10.0e6 freq

cell NOMINAL  bounds (-1.0e-7, 1.0e-7) action 0
cell WARN     bounds ( 5.0e-7, 2.0e-6) action 1   -- gap: 1e-7 .. 5e-7
cell CRITICAL bounds ( 2.0e-6, 1.0e-5) action 2

compose d=1 channels clock into stream

when NOMINAL  do emit ok
when WARM     do emit warn        -- typo: there is no cell 'WARM'
when CRITICAL do fire shutdown`,
    events: 400,
    noise: 0.4,
    expect: 'diagnostics',
  },

  // ── 04 ──────────────────────────────────────────────────────────────────────
  {
    id: 'composition',
    title: 'Composition & inflation',
    tagline: 'Many channels, one trajectory space — counted by T(n,d).',
    body: [
      'Real systems read several sources at once. `compose` binds d channels into a single trajectory space; each event is labelled by the channel it arrived on. The number of categorically distinguishable labelled trajectories over n events on d channels is not linear but combinatorial:',
      'T(n,d) = d · (1 + d)^(n−1).',
      'Adding a channel multiplies the state count by (1 + d) per event. This is the composition-inflation result — exponential discriminating power from pure timing, with no content transmitted. COMPILE shows the inflation table for the channel count you declared.',
      'Here two channels — a barometer and an IMU — compose into one altitude space. Change `d=2` and the channel list to add a third source and watch the T(n,d) table grow.',
    ],
    points: [
      'compose d=N channels … binds N channels into one trajectory space.',
      'T(n,d) = d·(1+d)^(n−1) — each event multiplies states by (1+d).',
      'COMPILE prints the inflation table; RUN shows per-channel activity.',
    ],
    script: `-- 04 · Composition & inflation
-- Two channels composed; T(n,d) counts distinguishable trajectories.

sync baro at 1.0e6 freq
sync imu  at 4.0e6 freq

cell ALT_LOW  bounds (-5.0e-6, -1.0e-6) action 0
cell ALT_HOLD bounds (-1.0e-6,  1.0e-6) action 1
cell ALT_HIGH bounds ( 1.0e-6,  5.0e-6) action 2

compose d=2 channels baro, imu into altitude_space

when ALT_LOW  do fire climb
when ALT_HOLD do emit hold
when ALT_HIGH do fire descend`,
    events: 600,
    noise: 0.45,
    expect: 'clean',
  },

  // ── 05 ──────────────────────────────────────────────────────────────────────
  {
    id: 'phase-machine',
    title: 'The phase machine',
    tagline: 'COMPILE and EXECUTE are mutually exclusive states.',
    body: [
      'A running trigger alternates between two phases. In COMPILE it accumulates events into an open trajectory — the cell is not yet determined. When the trajectory is complete it transitions to EXECUTE, where the composite cell is known and its action fires. The two phases never hold at once: an open trajectory has no cell to dispatch, a closed one is ready to fire.',
      'With d channels the trajectory length is d, so the engine spends d events accumulating (COMPILE) before each dispatch (EXECUTE). This program composes two channels, so the phase timeline alternates COMPILE → EXECUTE as pairs of events complete.',
      'RUN this and watch the PHASE TIMELINE strip: blue is COMPILE (accumulating), amber is EXECUTE (dispatching). The separation is structural, not a scheduling choice.',
    ],
    points: [
      'COMPILE accumulates; EXECUTE dispatches; they are mutually exclusive.',
      'Trajectory length equals the number of composed channels.',
      'Watch the PHASE TIMELINE strip alternate as trajectories close.',
    ],
    script: `-- 05 · The phase machine
-- Two channels → trajectory length 2 → COMPILE/EXECUTE alternate.

sync ch_a at 8.0e6 freq
sync ch_b at 8.0e6 freq

cell NEG bounds (-1.0e-6, 0.0)    action 0
cell POS bounds ( 0.0,    1.0e-6) action 1

compose d=2 channels ch_a, ch_b into pair

when NEG do emit neg
when POS do emit pos`,
    events: 500,
    noise: 0.5,
    expect: 'clean',
    feature: 'phase',
  },

  // ── 06 ──────────────────────────────────────────────────────────────────────
  {
    id: 'incorruptibility',
    title: 'Structural incorruptibility',
    tagline: 'No parser, no injection surface.',
    body: [
      'Because a pulse carries no content, there is nothing to parse — and a system with no parser has no injection surface. Buffer overflows, SQL injection, command injection: every one of these requires a payload that is interpreted as code. In Tempus the only thing that can satisfy a cell is a pulse with the right timing. An attacker can send any bytes they like; they have nowhere to go.',
      'The set of executable actions is fixed at compile time: it is exactly the actions named in the cell registry Γ, stored read-only. No runtime path leads from signal content to Γ. The attack surface is not reduced — it is absent by construction.',
      'This is a safety-critical coolant menu. COMPILE and RUN it, then use the injector below: paste arbitrary payloads — SQL, shellcode, JSON — and watch them get discarded. The count of executable actions |Cells(Γ)| never changes.',
    ],
    points: [
      'The action set is frozen at compile time = the cells in Γ.',
      'Arbitrary payloads have no parser to reach — they are discarded.',
      'Security is architectural, not a mitigation bolted on top.',
    ],
    script: `-- 06 · Structural incorruptibility
-- A safety-critical menu. There is no content parser anywhere.

sync coolant at 10.0e6 freq

cell NOMINAL  bounds (-1.0e-7, 1.0e-7) action 0
cell WARM     bounds ( 1.0e-7, 5.0e-7) action 1
cell HOT      bounds ( 5.0e-7, 2.0e-6) action 2
cell CRITICAL bounds ( 2.0e-6, 1.0e-5) action 3

compose d=1 channels coolant into coolant_traj

when NOMINAL  do emit status_ok
when WARM     do emit status_warn
when HOT      do begin
                 emit status_hot;
                 fire reduce_power
               end
when CRITICAL do begin
                 emit scram;
                 fire emergency_shutdown
               end`,
    events: 600,
    noise: 0.55,
    expect: 'clean',
    feature: 'inject',
  },
];

// ── Instrument examples ───────────────────────────────────────────────────────
// Each existing categorical-spectrometer instrument re-expressed as a Tempus
// program: its characteristic observable is read as a timing deviation ΔP, and
// its discrete measurement states become cells. Run one to see every value.

export const EXAMPLES: Lesson[] = [
  {
    id: 'thin-film',
    kind: 'example',
    title: 'Thin-film reflectance',
    tagline: 'Fabry–Pérot interference as a phase partition.',
    body: [
      'A dielectric film of thickness d on silicon reflects light with a round-trip phase δ = 2π·n₁·d/λ. The two-beam interference R(λ) is exactly the bounded-oscillator math the GPU shader runs in the Thin-Film instrument — and a bounded oscillator is a Tempus channel.',
      'Read the reflected wavefront’s deviation from the quarter-wave (destructive) reference as ΔP. Destructive interference (the antireflection null) sits at ΔP≈0; as δ drifts toward the half-wave condition the reflectance climbs to its constructive maximum. A film that is off-spec or contaminated lands on the far negative side.',
      'RUN it and read the Output tab: every pulse’s ΔP, the interference band it falls in, and the dispatched action — the full reflectance classification, not a single number.',
    ],
    points: [
      'δ = 2π·n₁·d/λ is the round-trip phase; ΔP is its deviation from quarter-wave.',
      'AR_NULL = destructive (antireflection); HIGH_R = constructive maximum.',
      'The same interference formula the instrument’s shader computes.',
    ],
    script: `-- Thin-film reflectance as a timing partition.
-- Round-trip phase δ = 2π·n1·d/λ ; ΔP = deviation from the quarter-wave null.

sync probe at 5.0e14 freq             -- optical carrier (~600 THz)

cell FOULED   bounds (-1.2e-6, -1.0e-7) action 3   -- film off-spec / contamination
cell AR_NULL  bounds (-1.0e-7,  1.0e-7) action 0   -- destructive: antireflection minimum
cell RISING   bounds ( 1.0e-7,  6.0e-7) action 1   -- partial reflection
cell HIGH_R   bounds ( 6.0e-7,  1.2e-6) action 2   -- constructive: reflectance maximum

compose d=1 channels probe into reflectance

when AR_NULL do emit antireflection_ok
when RISING  do emit partial_reflect
when HIGH_R  do emit high_reflectance
when FOULED  do fire flag_film`,
    events: 600,
    noise: 0.5,
    expect: 'clean',
  },
  {
    id: 'polymorphism',
    kind: 'example',
    title: 'Ritonavir polymorph ID',
    tagline: 'Form I vs Form II from the amide-I carbonyl shift.',
    body: [
      'Ritonavir’s Form I (cis amide) and Form II (trans amide) differ in the amide-I carbonyl frequency: ≈1695 cm⁻¹ for Form I, ≈1668 cm⁻¹ for Form II — the spectral signature the Polymorphism hologram resolves. Form II is the denser, less-soluble polymorph whose 1998 appearance pulled the drug from the market.',
      'Take the carbonyl peak position relative to the 1681 cm⁻¹ midpoint as ΔP. Positive ΔP → Form I (target); negative → Form II (reject); the overlapping middle band triggers a rescan rather than a guess.',
      'RUN it: the Output log lists each measurement’s ΔP and the polymorph cell it resolves to — a categorical identification with every value shown.',
    ],
    points: [
      'Amide-I carbonyl: Form I ≈1695, Form II ≈1668 cm⁻¹ (≈27 cm⁻¹ apart).',
      'ΔP = peak shift from the 1681 cm⁻¹ reference; sign selects the polymorph.',
      'AMBIGUOUS band dispatches a rescan, not a forced call.',
    ],
    script: `-- Ritonavir polymorph identification as a timing partition.
-- Amide-I carbonyl: Form I ~1695, Form II ~1668 cm-1.
-- ΔP = peak shift from the 1681 cm-1 reference.

sync ir_probe at 5.0e13 freq

cell FORM_II   bounds (-2.0e-6, -4.0e-7) action 2   -- trans amide ~1668: unwanted polymorph
cell AMBIGUOUS bounds (-4.0e-7,  4.0e-7) action 1   -- overlapping carbonyl band
cell FORM_I    bounds ( 4.0e-7,  2.0e-6) action 0   -- cis amide ~1695: target form

compose d=1 channels ir_probe into amide_I

when FORM_I    do emit identify_form_I
when AMBIGUOUS do emit rescan
when FORM_II   do begin
                  emit form_II_detected;
                  fire reject_lot
                end`,
    events: 500,
    noise: 0.4,
    expect: 'clean',
  },
  {
    id: 'bioreactor',
    kind: 'example',
    title: 'Bioreactor metabolic phase',
    tagline: 'ATP as the time coordinate; lactate/O₂ timing as ΔP.',
    body: [
      'In the Bioreactor instrument ATP is the time coordinate — evolution is dx/d[ATP]. The lactate/glucose flux oscillation and the dissolved-O₂ phase lag track the culture’s metabolic state, exactly the Michaelis–Menten amplitudes the metabolic hologram superposes.',
      'Two channels feed the trajectory: a lactate clock and an oxygen clock. Their combined timing deviation from the exponential-phase reference classifies the run — tight near zero in exponential growth, drifting positive as lactate accumulates and ATP falls through stationary phase into decline, and far out at metabolic collapse.',
      'This is a two-channel composition (d=2), so the engine accumulates a pair of events per trajectory. RUN it and watch the phase machine alternate while the Output log records every metabolic reading.',
    ],
    points: [
      'ATP is the clock; lactate + O₂ timing deviations are the datum.',
      'Cells track the growth curve: EXPONENTIAL → STATIONARY → DECLINE → CRITICAL.',
      'Two channels (d=2) → trajectory length 2; watch the phase strip.',
    ],
    script: `-- Bioreactor metabolic phase as a timing partition.
-- ATP is the time coordinate (dx/d[ATP]); lactate & O2 timing give ΔP.

sync lactate at 1.0e3 freq
sync oxygen  at 1.0e3 freq

cell EXPONENTIAL bounds (-1.0e-6, 1.0e-6) action 0   -- high ATP, low lactate
cell STATIONARY  bounds ( 1.0e-6, 4.0e-6) action 1   -- flux balanced
cell DECLINE     bounds ( 4.0e-6, 8.0e-6) action 2   -- lactate rising, ATP falling
cell CRITICAL    bounds ( 8.0e-6, 2.0e-5) action 3   -- metabolic collapse

compose d=2 channels lactate, oxygen into metabolic_state

when EXPONENTIAL do emit feed_nominal
when STATIONARY  do fire adjust_feed
when DECLINE     do begin
                    emit lactate_alarm;
                    fire harvest_soon
                  end
when CRITICAL    do fire emergency_harvest`,
    events: 600,
    noise: 0.5,
    expect: 'clean',
    feature: 'phase',
  },
  {
    id: 'ritonavir',
    kind: 'example',
    title: 'Synthesis QC trigger',
    tagline: 'A batch-release trigger for the 1998 Form II crisis.',
    body: [
      'The Synthesis instrument models 120 manufacturing batches in which cooling rate is the root cause of Form II contamination (faster, uncontrolled cooling crystallises the wrong polymorph). The release spec is Form II < 5% and purity > 95%.',
      'This is a textbook Tempus trigger: each batch’s crystallisation timing deviates from the controlled-cool reference by ΔP ∝ cooling rate. The partition is the QC decision — release in-spec lots, hold the marginal ones near the 5% Form II limit, reject out-of-spec, and quarantine a crash-cooled batch outright.',
      'RUN it to see a batch line classified pulse-by-pulse, with the accept/hold/reject/quarantine action and every ΔP in the Output log — the same accept-or-discard logic an LHC trigger applies to bunch crossings.',
    ],
    points: [
      'Cooling rate → Form II% is the causal link; ΔP ∝ cooling-rate deviation.',
      'Spec: Form II < 5%, purity > 95% → release; else hold / reject / quarantine.',
      'A real-time accept/reject trigger — the trigger pattern, on a synthesis line.',
    ],
    script: `-- Synthesis QC as a Tempus trigger (the 1998 Abbott Form II crisis).
-- Cooling rate drives Form II contamination; ΔP ∝ cooling-rate deviation.

sync crystalliser at 2.0e2 freq

cell IN_SPEC    bounds (-5.0e-7, 5.0e-7) action 0   -- Form II < 5%, purity > 95%
cell MARGINAL   bounds ( 5.0e-7, 1.5e-6) action 1   -- near the 5% Form II limit
cell OUT_SPEC   bounds ( 1.5e-6, 6.0e-6) action 2   -- Form II out of spec
cell CRASH_COOL bounds ( 6.0e-6, 1.5e-5) action 3   -- uncontrolled cool -> high Form II

compose d=1 channels crystalliser into batch_qc

when IN_SPEC    do emit release_lot
when MARGINAL   do fire hold_for_review
when OUT_SPEC   do fire reject_lot
when CRASH_COOL do begin
                   emit polymorph_alarm;
                   fire quarantine
                 end`,
    events: 600,
    noise: 0.45,
    expect: 'clean',
  },
  {
    id: 'psdr',
    kind: 'example',
    title: 'PSDR ensemble sync',
    tagline: 'Kuramoto phase-lock as an ensemble partition.',
    body: [
      'Phase-Synchronous Distributed Regulation runs an N-agent Kuramoto ensemble that locks when the coupling K exceeds the critical K_c. The order parameter r measures coherence; the Lyapunov V(S)=‖S−S*‖² deepens its basin as the agents synchronise — the partition the PSDR shader paints over [0,100]² state space.',
      'Each agent’s phase deviation from the mean field is its ΔP. Near zero the ensemble is phase-locked (r≈1); as deviations spread it passes through partial coherence and lagging drift into incoherence (K < K_c); a lone outlier on the far side is a rogue agent to eject.',
      'Three agents compose the ensemble (d=3), giving the largest composition-inflation state space in the set — check the Registry tab’s T(n,3) table. RUN it to read every agent’s phase deviation and the regulatory action it triggers.',
    ],
    points: [
      'ΔP = an agent’s phase deviation from the Kuramoto mean field.',
      'LOCKED (r≈1) → PARTIAL → INCOHERENT (K<K_c); ROGUE outliers ejected.',
      'd=3 ensemble — the richest T(n,d) state space here (see Registry).',
    ],
    script: `-- Phase-Synchronous Distributed Regulation as an ensemble partition.
-- Agents lock when K > K_c; ΔP = phase deviation from the mean field.

sync agent_a at 1.0e2 freq
sync agent_b at 1.0e2 freq
sync agent_c at 1.0e2 freq

cell ROGUE      bounds (-4.0e-6, -1.2e-6) action 3   -- outlier agent, eject
cell DRIFTING   bounds (-1.2e-6, -3.0e-7) action 1   -- lagging the mean field
cell LOCKED     bounds (-3.0e-7,  3.0e-7) action 0   -- r ~ 1, phase-locked
cell PARTIAL    bounds ( 3.0e-7,  1.2e-6) action 1   -- partial coherence
cell INCOHERENT bounds ( 1.2e-6,  4.0e-6) action 2   -- K < K_c, desynchronised

compose d=3 channels agent_a, agent_b, agent_c into ensemble

when LOCKED     do emit nominal
when PARTIAL    do emit converging
when DRIFTING   do fire nudge_phase
when INCOHERENT do fire raise_coupling
when ROGUE      do fire eject_agent`,
    events: 600,
    noise: 0.55,
    expect: 'clean',
  },
];
