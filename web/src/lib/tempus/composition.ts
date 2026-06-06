// ── Composition-trajectory → atom engine ──────────────────────────────────────
//
// The "better method": a script IS a labeled composition (composition-inflation
// .tex). n cycles in d-dim S-entropy give T(n,d) = d·(d+1)^(n-1) reachable
// trajectories; this one walks to a partition state (n,ℓ,m,s) = a part of an
// atom. We make the mapping controllable so a script BUILDS a named element:
//   Sk ticks  → principal shell n   (more Sk ⇒ deeper shell)
//   St ticks  → angular ℓ (orbital: 0=s, 1=p, 2=d, …)
//   Se ticks  → orientation m
//   spin +/-  → parity s
// The partition index of (n,ℓ,m,s) is the atomic number Z → a real element
// (shell 2 = Li…Ne, exactly the models in /public/atoms). The composition
// compiles into an atom; manipulate it → a different atom.
//
//   space d=3 nmax=8
//   refine Sk 1        -- shell 2
//   refine St 1        -- p orbital
//   spin +             -- → Carbon (Z=6)

import type { Diag } from './types';

export const AXES = ['Sk', 'St', 'Se'] as const;
export type Axis = (typeof AXES)[number];
const AXIS_INDEX: Record<string, number> = { sk: 0, st: 1, se: 2, s1: 0, s2: 1, s3: 2 };

export interface Segment { axis: Axis; count: number; }
export interface SPoint { sk: number; st: number; se: number; }

export interface Element {
  Z: number; symbol: string; name: string; config: string; model: string | null;
}

export interface PartitionState { n: number; l: number; m: number; s: number; label: string; }

export interface CompositionScene {
  d: number;
  nmax: number;
  segments: Segment[];
  spin: number;              // +0.5 / -0.5
  n: number;                 // total cycles = Σ counts
  T: number;                 // composition-inflation count
  dTheta: number;            // 2π/T
  bits: number;              // log2 T
  path: SPoint[];            // trajectory through [0,1]³ (converging walk)
  coord: SPoint;             // resolved S-entropy coordinate
  state: PartitionState;     // resolved partition state
  element: Element;          // the atom the composition builds
}

export interface CompositionResult { ok: boolean; diagnostics: Diag[]; scene: CompositionScene | null; }

const MAX_N = 24;
const L_LETTERS = ['s', 'p', 'd', 'f', 'g', 'h', 'i'];

export function inflation(n: number, d: number): number { return d * Math.pow(d + 1, n - 1); }

// ── periodic table (first 18 + a few), with the GLB models present on disk ────
export const ELEMENTS: Element[] = [
  { Z: 1, symbol: 'H', name: 'Hydrogen', config: '1s¹', model: null },
  { Z: 2, symbol: 'He', name: 'Helium', config: '1s²', model: null },
  { Z: 3, symbol: 'Li', name: 'Lithium', config: '[He] 2s¹', model: 'lithium.glb' },
  { Z: 4, symbol: 'Be', name: 'Beryllium', config: '[He] 2s²', model: 'berlylium.glb' },
  { Z: 5, symbol: 'B', name: 'Boron', config: '[He] 2s² 2p¹', model: 'boron.glb' },
  { Z: 6, symbol: 'C', name: 'Carbon', config: '[He] 2s² 2p²', model: 'carbon.glb' },
  { Z: 7, symbol: 'N', name: 'Nitrogen', config: '[He] 2s² 2p³', model: 'nitrogen.glb' },
  { Z: 8, symbol: 'O', name: 'Oxygen', config: '[He] 2s² 2p⁴', model: 'oxygen.glb' },
  { Z: 9, symbol: 'F', name: 'Fluorine', config: '[He] 2s² 2p⁵', model: 'flourine.glb' },
  { Z: 10, symbol: 'Ne', name: 'Neon', config: '[He] 2s² 2p⁶', model: 'neon_atom.glb' },
  { Z: 11, symbol: 'Na', name: 'Sodium', config: '[Ne] 3s¹', model: null },
  { Z: 12, symbol: 'Mg', name: 'Magnesium', config: '[Ne] 3s²', model: null },
  { Z: 13, symbol: 'Al', name: 'Aluminium', config: '[Ne] 3s² 3p¹', model: null },
  { Z: 14, symbol: 'Si', name: 'Silicon', config: '[Ne] 3s² 3p²', model: null },
  { Z: 15, symbol: 'P', name: 'Phosphorus', config: '[Ne] 3s² 3p³', model: null },
  { Z: 16, symbol: 'S', name: 'Sulfur', config: '[Ne] 3s² 3p⁴', model: null },
  { Z: 17, symbol: 'Cl', name: 'Chlorine', config: '[Ne] 3s² 3p⁵', model: null },
  { Z: 18, symbol: 'Ar', name: 'Argon', config: '[Ne] 3s² 3p⁶', model: null },
];

const Nstate = (k: number): number => (k <= 0 ? 0 : (k * (k + 1) * (2 * k + 1)) / 3);

// partition index (atomic number) of a state, via the bijection's lexicographic
// enumeration within each shell: N_state(n-1) + offset.
function partitionIndex(n: number, l: number, m: number, s: number): number {
  let off = 0;
  for (let lp = 0; lp < l; lp++) off += 2 * (2 * lp + 1);
  off += 2 * (m + l) + (s > 0 ? 2 : 1);
  return Nstate(n - 1) + off;
}

function elementOf(Z: number): Element {
  return ELEMENTS.find(e => e.Z === Z) ?? { Z, symbol: `Z${Z}`, name: `Element ${Z}`, config: '—', model: null };
}

export function compileComposition(src: string): CompositionResult {
  const diagnostics: Diag[] = [];
  const segments: Segment[] = [];
  let d = 3, nmax = 8, spin = 0.5;

  const lines = src.split('\n');
  let offset = 0;
  for (const raw of lines) {
    const lineStart = offset;
    offset += raw.length + 1;
    const line = raw.replace(/--.*$/, '').trim();
    if (!line) continue;
    const err = (message: string) => diagnostics.push({ severity: 'error', message, pos: lineStart });
    const head = line.split(/\s+/)[0].toLowerCase();

    if (head === 'space') {
      const dm = line.match(/d\s*=\s*(\d+)/i);
      const nm = line.match(/nmax\s*=\s*(\d+)/i);
      if (dm) d = parseInt(dm[1], 10);
      if (nm) nmax = parseInt(nm[1], 10);
      if (d < 1 || d > 6) { err(`space: d must be in 1..6`); d = 3; }
      if (nmax < 2 || nmax > 32) { err(`space: nmax must be in 2..32`); nmax = 8; }

    } else if (head === 'spin') {
      const m = line.match(/^spin\s+([+-]|up|down)/i);
      if (!m) { err(`spin: expected 'spin +' or 'spin -'`); continue; }
      spin = /[-]|down/i.test(m[1]) ? -0.5 : 0.5;

    } else if (head === 'refine') {
      const m = line.match(/^refine\s+(\S+)\s+(\d+)/i);
      if (!m) { err(`refine: expected 'refine <axis> <count>' (axis ∈ Sk, St, Se)`); continue; }
      const axisKey = m[1].toLowerCase();
      if (!(axisKey in AXIS_INDEX)) { err(`refine: unknown axis '${m[1]}' (use Sk, St, Se)`); continue; }
      const idx = AXIS_INDEX[axisKey];
      if (idx >= d) { err(`refine: axis '${m[1]}' exceeds d=${d}`); continue; }
      const count = parseInt(m[2], 10);
      if (count < 1) { err(`refine: count must be ≥ 1`); continue; }
      segments.push({ axis: AXES[idx], count });

    } else {
      err(`unknown directive '${head}' (expected: space, refine, spin)`);
    }
  }

  if (segments.length === 0) diagnostics.push({ severity: 'error', message: 'no segments: declare at least one `refine <axis> <count>`' });
  const nTotal = segments.reduce((a, s) => a + s.count, 0);
  if (nTotal > MAX_N) diagnostics.push({ severity: 'error', message: `total cycles n=${nTotal} exceeds the ${MAX_N}-cycle browser limit` });

  const ok = !diagnostics.some(x => x.severity === 'error');
  if (!ok) return { ok, diagnostics, scene: null };

  // ── resolve the partition state from per-axis dwell counts ─────────────────
  const cnt = [0, 0, 0];
  for (const seg of segments) cnt[AXIS_INDEX[seg.axis.toLowerCase()]] += seg.count;
  const n = Math.min(nmax, 1 + cnt[0]);                       // Sk → shell
  const l = Math.min(n - 1, cnt[1]);                          // St → orbital ℓ
  const m = l > 0 ? (cnt[2] % (2 * l + 1)) - l : 0;           // Se → orientation
  const s = spin;
  const Z = partitionIndex(n, l, m, s);
  const element = elementOf(Z);
  const letter = L_LETTERS[l] ?? `(l=${l})`;
  const state: PartitionState = { n, l, m, s, label: `${n}${letter}${m !== 0 ? `(m=${m > 0 ? '+' : ''}${m})` : ''}` };

  // resolved S-entropy coordinate (forward embedding) + a converging walk for view
  const coord: SPoint = {
    sk: (n - 1) / (nmax - 1),
    st: n > 1 ? l / (n - 1) : 0,
    se: l > 0 ? (m + l) / (2 * l) : 0.5,
  };
  const path: SPoint[] = [];
  const cur = { sk: 0.5, st: 0.5, se: 0.5 };
  const tgt = [coord.sk, coord.st, coord.se];
  for (const seg of segments) {
    const a = AXIS_INDEX[seg.axis.toLowerCase()];
    for (let t = 0; t < seg.count; t++) {
      const k = a === 0 ? 'sk' : a === 1 ? 'st' : 'se';
      cur[k] += (tgt[a] - cur[k]) * 0.6;
      path.push({ sk: cur.sk, st: cur.st, se: cur.se });
    }
  }

  const T = inflation(nTotal, d);
  return {
    ok: true, diagnostics,
    scene: { d, nmax, segments, spin, n: nTotal, T, dTheta: (2 * Math.PI) / T, bits: Math.log2(T), path, coord, state, element },
  };
}
