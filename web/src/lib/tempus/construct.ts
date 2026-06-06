// ── Tempus construction DSL ───────────────────────────────────────────────────
//
// A second Tempus surface for *constructing* items rather than classifying
// timing events. An item is built from oscillator modes (a spectrum); rendering
// the superposition on the GPU IS the computation (per-pixel interference =
// observation). This module parses the DSL into a Scene that the WebGL
// interference engine renders. Pure TypeScript, line-oriented, forgiving.
//
//   domain 40mm x 40mm
//   wave a at (-8mm, 0) wavelength 6mm amplitude 1 phase 0
//   wave b at ( 8mm, 0) wavelength 6mm
//   slits 5 spacing 6mm wavelength 6mm        -- macro: N coherent sources
//   render
//   observe visibility, fringes

import type { Diag } from './types';

export interface Wave {
  x: number; y: number;      // source position, metres
  amp: number;               // amplitude
  phase: number;             // radians
  wavelength: number;        // metres
}

export interface ConstructScene {
  domain: { w: number; h: number };   // metres
  waves: Wave[];
  observes: string[];
  hasRender: boolean;
}

export interface ConstructResult {
  ok: boolean;
  diagnostics: Diag[];
  scene: ConstructScene | null;
}

const MAX_WAVES = 32;
const KNOWN_OBSERVES = ['visibility', 'fringes', 'sources', 'intensity_max', 'intensity_min'];

// ── length / phase parsing ────────────────────────────────────────────────────
const UNIT: Record<string, number> = { m: 1, cm: 1e-2, mm: 1e-3, um: 1e-6, 'µm': 1e-6, nm: 1e-9 };

function parseLen(tok: string): number | null {
  const m = tok.match(/^(-?\d*\.?\d+(?:e[+-]?\d+)?)\s*(m|cm|mm|um|µm|nm)?$/i);
  if (!m) return null;
  const v = parseFloat(m[1]);
  const u = m[2] ? UNIT[m[2].toLowerCase()] : 1; // bare number → metres
  return v * (u ?? 1);
}

function parseNum(tok: string): number | null {
  // plain number, optionally suffixed by `pi` (→ ×π)
  const m = tok.match(/^(-?\d*\.?\d+(?:e[+-]?\d+)?)\s*(pi|π)?$/i);
  if (!m) return null;
  return parseFloat(m[1]) * (m[2] ? Math.PI : 1);
}

// ── compile ───────────────────────────────────────────────────────────────────
export function compileConstruct(src: string): ConstructResult {
  const diagnostics: Diag[] = [];
  const waves: Wave[] = [];
  let domain = { w: 0.04, h: 0.04 };   // default 40mm × 40mm
  let observes: string[] = [];
  let hasRender = false;
  const names = new Set<string>();

  const lines = src.split('\n');
  let offset = 0;
  for (const raw of lines) {
    const lineStart = offset;
    offset += raw.length + 1;
    const line = raw.replace(/--.*$/, '').trim();   // strip comment
    if (!line) continue;
    const err = (message: string) => diagnostics.push({ severity: 'error', message, pos: lineStart });
    const warn = (message: string) => diagnostics.push({ severity: 'warning', message, pos: lineStart });

    const head = line.split(/\s+/)[0].toLowerCase();

    if (head === 'domain') {
      const m = line.match(/^domain\s+(\S+)\s*x\s*(\S+)/i);
      const w = m && parseLen(m[1]);
      const h = m && parseLen(m[2]);
      if (!m || w == null || h == null) { err(`domain: expected 'domain <W> x <H>' (e.g., domain 40mm x 40mm)`); continue; }
      domain = { w, h };

    } else if (head === 'wave') {
      const m = line.match(/^wave\s+(\S+)\s+at\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)/i);
      if (!m) { err(`wave: expected 'wave <name> at (<x>, <y>) wavelength <λ> [amplitude <a>] [phase <p>]'`); continue; }
      const name = m[1];
      if (names.has(name)) warn(`wave '${name}' redeclared`);
      names.add(name);
      const x = parseLen(m[2]), y = parseLen(m[3]);
      const lamTok = line.match(/wavelength\s+(\S+)/i);
      const lam = lamTok && parseLen(lamTok[1]);
      const ampTok = line.match(/amplitude\s+(\S+)/i);
      const phTok = line.match(/phase\s+(\S+)/i);
      if (x == null || y == null) { err(`wave '${name}': bad position`); continue; }
      if (!lamTok || lam == null || lam <= 0) { err(`wave '${name}': missing or invalid 'wavelength'`); continue; }
      const amp = ampTok ? parseNum(ampTok[1]) : 1;
      const phase = phTok ? parseNum(phTok[1]) : 0;
      if (amp == null) { err(`wave '${name}': bad amplitude`); continue; }
      if (phase == null) { err(`wave '${name}': bad phase`); continue; }
      waves.push({ x, y, amp, phase, wavelength: lam });

    } else if (head === 'slits') {
      const m = line.match(/^slits\s+(\d+)\s+spacing\s+(\S+)\s+wavelength\s+(\S+)/i);
      if (!m) { err(`slits: expected 'slits <N> spacing <d> wavelength <λ> [amplitude <a>] [at_y <y>]'`); continue; }
      const N = parseInt(m[1], 10);
      const d = parseLen(m[2]);
      const lam = parseLen(m[3]);
      const ampTok = line.match(/amplitude\s+(\S+)/i);
      const yTok = line.match(/at_y\s+(\S+)/i);
      const amp = ampTok ? parseNum(ampTok[1]) : 1;
      const y = yTok ? parseLen(yTok[1]) : 0;
      if (N < 1) { err(`slits: N must be ≥ 1`); continue; }
      if (d == null || lam == null || lam <= 0 || amp == null || y == null) { err(`slits: bad parameter`); continue; }
      for (let i = 0; i < N; i++) {
        const x = (i - (N - 1) / 2) * d;
        waves.push({ x, y, amp, phase: 0, wavelength: lam });
      }

    } else if (head === 'render') {
      hasRender = true;

    } else if (head === 'observe') {
      const items = line.slice(7).split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
      for (const it of items) {
        if (!KNOWN_OBSERVES.includes(it)) diagnostics.push({ severity: 'info', message: `observe '${it}': unknown metric (known: ${KNOWN_OBSERVES.join(', ')})`, pos: lineStart });
      }
      observes = observes.concat(items);

    } else {
      err(`unknown directive '${head}' (expected: domain, wave, slits, render, observe)`);
    }
  }

  // ── whole-program checks ────────────────────────────────────────────────────
  if (waves.length === 0) diagnostics.push({ severity: 'error', message: 'no sources: declare a `wave` or a `slits` macro' });
  if (waves.length > MAX_WAVES) diagnostics.push({ severity: 'warning', message: `${waves.length} sources exceeds the ${MAX_WAVES}-source GPU limit; extra sources are ignored` });
  if (!hasRender && waves.length > 0) diagnostics.push({ severity: 'warning', message: 'no `render` directive — add `render` to compute the interference field' });
  if (observes.length === 0 && waves.length > 0) observes = ['visibility', 'fringes', 'sources'];

  const ok = !diagnostics.some(d => d.severity === 'error');
  const scene: ConstructScene | null = ok
    ? { domain, waves: waves.slice(0, MAX_WAVES), observes, hasRender }
    : null;
  return { ok, diagnostics, scene };
}
