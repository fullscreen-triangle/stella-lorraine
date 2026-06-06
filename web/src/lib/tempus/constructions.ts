// ── Interference Lab — construction-track lessons ─────────────────────────────
// A second tutorial track. Instead of classifying timing events, these scripts
// CONSTRUCT an item from oscillator modes (a spectrum) and let the GPU render
// the superposition — per-pixel interference IS the computation. Start from
// well-known physics experiments and build toward spectral construction.
// These use the construction DSL (construct.ts), not the timing language.

import type { Lesson } from './lessons';

// ── Atom construction by composition trajectory (composition.ts) ──────────────
// A script IS a labeled composition. Walk it through S-entropy space → land on a
// partition state (n,ℓ,m,s) = a part of an atom. Manipulate the composition →
// a different part. T(n,3) = 3·4^(n-1) parts are reachable.

export const ATOMS: Lesson[] = [
  {
    id: 'build-carbon',
    kind: 'lab',
    lang: 'compose',
    title: 'Compile an atom: Carbon',
    tagline: 'The composition compiles into an element — rendered in 3D.',
    body: [
      'The better method, made literal: the script is a labeled composition, and it compiles into an atom. Sk ticks set the principal shell n, St ticks set the orbital ℓ (0=s, 1=p, …), Se ticks and spin set the rest. The resolved (n,ℓ,m,s) has an atomic number Z — a real element.',
      'Below: one Sk tick → shell 2; one St tick → a p-orbital; spin + → atomic number 6. RUN it and open the Atom domain — the output is Carbon, rendered as its 3D model. The Inflation domain shows how many elements this depth can reach (T(n,3) = 3·4^(n-1)).',
      'No wave sources, no interference. You wrote a composition; the GPU rendered an atom.',
    ],
    points: [
      'Sk → shell n · St → orbital ℓ · Se + spin → m, s.',
      'Resolved (n,ℓ,m,s) → atomic number Z → a 3D element model.',
      'Open the Atom domain to see (and orbit) the rendered atom.',
    ],
    script: `-- Compile Carbon (Z=6): shell 2, a p-orbital.
space d=3 nmax=8

refine Sk 1      -- shell n = 2
refine St 1      -- p orbital (ℓ = 1)
spin +           -- → Carbon`,
    events: 0, noise: 0, expect: 'clean',
  },
  {
    id: 'walk-the-shell',
    kind: 'lab',
    lang: 'compose',
    title: 'Walk shell 2: Li → Ne',
    tagline: 'Manipulate the composition to build a different element.',
    body: [
      'Shell 2 holds eight partition states — exactly Lithium through Neon. Manipulate the composition and you walk across them: add St to enter the p-orbitals, add Se to step the orientation m, flip the spin to reach the neighbouring element.',
      'This script lands on Oxygen (Z=8). Change `refine Se 1` to `Se 2`, or toggle `spin +`/`spin -`, or drop St to fall back to the s-block (Li, Be), then RUN — the Atom domain swaps to the new element’s 3D model.',
      'Eight states, eight elements, all reachable by editing the composition. That is composition-inflation building the second row of the periodic table.',
    ],
    points: [
      'Shell 2 = Li, Be, B, C, N, O, F, Ne (its 8 partition states).',
      'St enters the p-block; Se steps m; spin reaches the neighbour.',
      'Each edit re-resolves Z → a different 3D atom.',
    ],
    script: `-- Build Oxygen (Z=8). Edit Se / spin to walk Li → Ne.
space d=3 nmax=8

refine Sk 1      -- shell 2
refine St 1      -- p-block
refine Se 1      -- orientation m
spin +           -- → Oxygen`,
    events: 0, noise: 0, expect: 'clean',
  },
  {
    id: 's-block',
    kind: 'lab',
    lang: 'compose',
    title: 'The s-block: spin picks the element',
    tagline: 'No St → an s-orbital; spin selects Lithium or Beryllium.',
    body: [
      'Drop the angular axis entirely (no St) and the trajectory stays in the s-orbital (ℓ=0). Shell 2 with ℓ=0 is the s-block: Lithium and Beryllium. The only freedom left is spin, which selects between them.',
      'This builds Lithium (spin −). Flip to `spin +` and RUN — the Atom domain swaps to Beryllium. Two electrons, two elements, one bit of difference: the parity s.',
      'This is the smallest possible manipulation that changes the atom — a single spin flip.',
    ],
    points: [
      'No St ⇒ ℓ=0 (s-orbital); shell 2 s-block = Li, Be.',
      'spin − → Lithium, spin + → Beryllium.',
      'The parity s is the final bit that names the element.',
    ],
    script: `-- s-block of shell 2. Flip spin: Li ↔ Be.
space d=3 nmax=8

refine Sk 1      -- shell 2, s-orbital (no St)
spin -           -- → Lithium  (try: spin +)`,
    events: 0, noise: 0, expect: 'clean',
  },
  {
    id: 'deeper-shell',
    kind: 'lab',
    lang: 'compose',
    title: 'Deeper: more Sk → shell 3',
    tagline: 'Each Sk tick drives one shell deeper.',
    body: [
      'Sk is the shell axis: each Sk tick adds one to the principal shell n. Two Sk ticks → shell 3, the third row (Na, Mg, … Ar). This script lands on Sodium (Z=11).',
      'Shell 3 elements have no 3D model in this build, so the Atom domain falls back to a shell-ring diagram — but the resolution is exact, and the Inflation domain shows the reachable count climbing as you add cycles (each cycle ×4).',
      'Add another `refine Sk 1` to reach shell 4, or add St/Se to move along the row. The composition keeps building deeper atoms.',
    ],
    points: [
      'n = 1 + (Sk ticks); two Sk ticks → shell 3.',
      'Shell 3 = Na…Ar (no GLB model here → shell diagram).',
      'Composition-inflation: each extra cycle ×4 reachable atoms.',
    ],
    script: `-- Two Sk ticks → shell 3 → Sodium (Z=11).
space d=3 nmax=8

refine Sk 2      -- shell n = 3
spin -           -- → Sodium`,
    events: 0, noise: 0, expect: 'clean',
  },
];

export const LAB: Lesson[] = [
  {
    id: 'double-slit',
    kind: 'lab',
    lang: 'construct',
    title: "Young's double slit",
    tagline: 'Two coherent oscillators interfere — the canonical experiment.',
    body: [
      'The simplest constructed item: two point sources oscillating at the same wavelength and phase. Where their wavefronts meet in step they add (bright); where they meet out of step they cancel (dark). The result is the interference field — the striped pattern Young measured in 1801.',
      'There is no simulation here. The fragment shader evaluates the superposition Σ A·cos(k·r + φ) at every pixel in parallel; by the Observation–Computation Equivalence, that rendered field IS the measurement. The GPU is the instrument.',
      'Press COMPILE to check the scene, then RUN to render it. Read the measured fringe visibility and count below. Then move a source, change the wavelength, and watch the fringes respond.',
    ],
    points: [
      'An item is built from oscillator modes — here, two sources.',
      'Rendering the superposition = computing the interference (no simulation).',
      'Visibility ≈ 1 for two equal, coherent sources.',
    ],
    script: `-- Young's double slit: two coherent point sources.
domain 40mm x 40mm

wave s1 at (-8mm, 0) wavelength 6mm amplitude 1 phase 0
wave s2 at ( 8mm, 0) wavelength 6mm amplitude 1 phase 0

render
observe visibility, fringes, sources`,
    events: 0, noise: 0, expect: 'clean',
  },
  {
    id: 'grating',
    kind: 'lab',
    lang: 'construct',
    title: 'Diffraction grating',
    tagline: 'Many coherent sources sharpen the maxima.',
    body: [
      'Add more sources in a line and the broad two-slit fringes collapse into narrow, bright principal maxima separated by dark gaps — a diffraction grating. The `slits` macro lays down N equally spaced coherent sources for you.',
      'This is construction by repetition: the same oscillator mode, copied N times across space. The more copies, the sharper the constructed pattern — the grating resolves wavelength better as N grows.',
      'RUN it, then change `slits 6` to `slits 12` and re-run. The principal maxima get narrower while their spacing stays fixed. Read how the measured fringe count tracks N.',
    ],
    points: [
      '`slits N spacing d wavelength λ` → N coherent sources.',
      'More sources → sharper principal maxima (better resolution).',
      'Spacing of maxima is set by d/λ, sharpness by N.',
    ],
    script: `-- Diffraction grating: N coherent sources in a row.
domain 60mm x 40mm

slits 6 spacing 6mm wavelength 6mm amplitude 1

render
observe visibility, fringes, sources`,
    events: 0, noise: 0, expect: 'clean',
  },
  {
    id: 'phase-steer',
    kind: 'lab',
    lang: 'construct',
    title: 'Manipulating the field',
    tagline: 'Shift one source’s phase — steer the whole pattern.',
    body: [
      'Constructing an item is only half the story; you also manipulate it. The phase of a source is a control knob: advance one source by π and every bright fringe moves to where a dark one was. The entire interference field translates.',
      'This is phased-array steering — the principle behind beam-forming radar and ultrasound. In the framework it is a partition-coordinate manipulation: you are editing the spectrum (the phase of a mode) and the rendered item updates accordingly.',
      'RUN it, then change `phase 1pi` on s2 to `phase 0.5pi` and re-run to slide the fringes by a quarter period. Phase is written in units of π (so `1pi` = π radians).',
    ],
    points: [
      'Phase is a control: Δφ on one source translates the whole field.',
      'phase is given in units of π — `0.5pi`, `1pi`, `2pi`.',
      'This is phased-array / beam steering, expressed as a spectrum edit.',
    ],
    script: `-- Manipulate the item: shift one source's phase by π.
domain 40mm x 40mm

wave s1 at (-8mm, 0) wavelength 6mm phase 0
wave s2 at ( 8mm, 0) wavelength 6mm phase 1pi

render
observe visibility, fringes`,
    events: 0, noise: 0, expect: 'clean',
  },
  {
    id: 'visibility',
    kind: 'lab',
    lang: 'construct',
    title: 'Visibility is the match',
    tagline: 'Mismatched spectra wash the fringes out.',
    body: [
      'Two sources at the SAME wavelength produce crisp, high-visibility fringes. Give them DIFFERENT wavelengths and their fringe systems drift out of register; averaged together the contrast collapses and the visibility falls toward zero.',
      'This is the heart of universal spectral matching: the visibility of an interference pattern measures how well two spectra agree. Constructive overlap = match; washed-out fringes = mismatch. Comparing any two items — molecules, signals, images — reduces to rendering this one number.',
      'RUN it and read the lowered visibility against the double-slit lesson. Then bring `b`’s wavelength back toward 6mm and watch the fringes — and the match — sharpen.',
    ],
    points: [
      'Equal wavelengths → high visibility (match); unequal → low (mismatch).',
      'Visibility is the single number behind all spectral matching.',
      'Switch the Render DOMAIN: Frequency shows the two peaks, Time shows their beat — the same item re-projected (mean-recovery invariant).',
    ],
    script: `-- Visibility = the match. Two different wavelengths lose coherence.
domain 40mm x 40mm

wave a at (-8mm, 0) wavelength 6mm
wave b at ( 8mm, 0) wavelength 9mm

render
observe visibility, fringes`,
    events: 0, noise: 0, expect: 'clean',
  },
];
