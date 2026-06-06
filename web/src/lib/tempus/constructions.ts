// ── Interference Lab — construction-track lessons ─────────────────────────────
// A second tutorial track. Instead of classifying timing events, these scripts
// CONSTRUCT an item from oscillator modes (a spectrum) and let the GPU render
// the superposition — per-pixel interference IS the computation. Start from
// well-known physics experiments and build toward spectral construction.
// These use the construction DSL (construct.ts), not the timing language.

import type { Lesson } from './lessons';

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
      'Constructive = match, destructive = mismatch — one render pass.',
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
