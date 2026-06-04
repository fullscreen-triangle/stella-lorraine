// ── Tempus static semantics (the COMPILE pass) ────────────────────────────────
//
// Implements the well-formedness checks of the Temporal Programming paper
// (§3.2): every `when` references a declared cell, every `compose` references
// declared sync sources, and the cell partition tiles ΔP-space. Cell coverage
// and disjointness are surfaced as diagnostics rather than silently accepted,
// so a learner can see exactly what the compiler accepts before it freezes the
// cell registry Γ. Pure TypeScript — no runtime side effects.

import { lex }          from './lexer';
import { parse }        from './parser';
import { buildRuntime } from './runtime';
import type { CellDecl, Diag, ParsedProgram, Program } from './types';

export interface RegistrySummary {
  cells:    CellDecl[];      // compiled cells, in declaration order
  channels: string[];        // active channels (compose, else sync sources)
  span:     [number, number] | null;   // [min lo, max hi] of the partition
  /** Composition-inflation T(n,d) = d·(1+d)^(n-1) for the compiled channel count. */
  inflation: { d: number; rows: { n: number; T: number }[] } | null;
}

export interface CompileResult {
  ok:          boolean;          // false iff any diagnostic is an error
  diagnostics: Diag[];
  program:     Program;          // raw AST (always returned)
  runtime:     ParsedProgram | null;   // compiled registry, when buildable
  registry:    RegistrySummary | null;
}

/** Composition-inflation count T(n,d) = d·(1+d)^(n-1). */
export function inflation(n: number, d: number): number {
  return d * Math.pow(1 + d, n - 1);
}

export function compile(src: string): CompileResult {
  const diagnostics: Diag[] = [];

  const tokens  = lex(src);
  const program = parse(tokens, diagnostics);   // collects syntax errors

  // ── Duplicate cell names (the Map in buildRuntime would silently coalesce) ──
  const seen = new Set<string>();
  for (const d of program.decls) {
    if (d.kind === 'cell') {
      if (seen.has(d.name)) {
        diagnostics.push({ severity: 'error', message: `duplicate cell '${d.name}': a cell name must be unique in the registry` });
      }
      seen.add(d.name);
    }
  }

  // ── Build the runtime registry (may throw if no cells) ─────────────────────
  let runtime: ParsedProgram | null = null;
  try {
    runtime = buildRuntime(program);
  } catch (e: unknown) {
    diagnostics.push({ severity: 'error', message: e instanceof Error ? e.message : String(e) });
  }

  let registry: RegistrySummary | null = null;

  if (runtime) {
    const cells    = Array.from(runtime.cells.values());
    const channels = runtime.compose?.channels ?? Array.from(runtime.syncs.keys());

    // (1) cell bounds must be non-empty intervals -----------------------------
    for (const c of cells) {
      if (!(c.lo < c.hi)) {
        diagnostics.push({ severity: 'error', message: `cell '${c.name}' has empty bounds (${c.lo}, ${c.hi}): lo must be < hi` });
      }
    }

    // (2) every `when` must reference a declared cell -------------------------
    for (const cellName of Array.from(runtime.whens.keys())) {
      if (!runtime.cells.has(cellName)) {
        const hint = nearest(cellName, Array.from(runtime.cells.keys()));
        diagnostics.push({
          severity: 'error',
          message: `when ${cellName}: undeclared cell` + (hint ? ` (did you mean '${hint}'?)` : ''),
        });
      }
    }

    // (3) every composed channel must be a declared sync source ---------------
    if (runtime.compose) {
      for (const ch of runtime.compose.channels) {
        if (!runtime.syncs.has(ch)) {
          const hint = nearest(ch, Array.from(runtime.syncs.keys()));
          diagnostics.push({
            severity: 'error',
            message: `compose channel '${ch}': no matching sync source` + (hint ? ` (did you mean '${hint}'?)` : ''),
          });
        }
      }
      // declared arity d=n should match the channel list length
      if (runtime.compose.d !== runtime.compose.channels.length) {
        diagnostics.push({
          severity: 'warning',
          message: `compose declares d=${runtime.compose.d} but lists ${runtime.compose.channels.length} channel(s)`,
        });
      }
    }

    // (4) at least one channel must exist -------------------------------------
    if (channels.length === 0) {
      diagnostics.push({ severity: 'error', message: 'no channels: declare a `sync` source (and reference it in `compose`)' });
    }

    // (5) coverage + disjointness of the ΔP partition -------------------------
    const span = coverageChecks(cells, diagnostics);

    // (6) unused cells (declared but never dispatched) — informational --------
    for (const c of cells) {
      if (c.lo < c.hi && !runtime.whens.has(c.name)) {
        diagnostics.push({ severity: 'info', message: `cell '${c.name}' has no 'when' rule — it classifies events but dispatches nothing` });
      }
    }

    const d = channels.length;
    registry = {
      cells,
      channels,
      span,
      inflation: d >= 1
        ? { d, rows: [1, 2, 3, 4, 5, 6].map(n => ({ n, T: inflation(n, d) })) }
        : null,
    };
  }

  const ok = !diagnostics.some(x => x.severity === 'error');
  return { ok, diagnostics, program, runtime, registry };
}

// ── Coverage / disjointness over the global cell partition ────────────────────
function coverageChecks(cells: CellDecl[], diagnostics: Diag[]): [number, number] | null {
  const valid = cells.filter(c => c.lo < c.hi).slice().sort((a, b) => a.lo - b.lo);
  if (valid.length === 0) return null;

  const span: [number, number] = [valid[0].lo, valid[0].hi];
  for (let i = 1; i < valid.length; i++) {
    const prev = valid[i - 1];
    const cur  = valid[i];
    span[1] = Math.max(span[1], cur.hi);

    if (cur.lo > prev.hi + 1e-18) {
      // gap: ΔP values in (prev.hi, cur.lo) fall to the anomaly action
      diagnostics.push({
        severity: 'warning',
        message: `coverage gap in [${fmt(prev.hi)}, ${fmt(cur.lo)}] between '${prev.name}' and '${cur.name}' — ΔP here dispatches the anomaly action`,
      });
    } else if (cur.lo < prev.hi - 1e-18) {
      // genuine overlap (beyond a shared measure-zero boundary)
      diagnostics.push({
        severity: 'warning',
        message: `cells '${prev.name}' and '${cur.name}' overlap on [${fmt(cur.lo)}, ${fmt(prev.hi)}] — first match wins`,
      });
    }
  }
  return span;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmt(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 1e-6) return `${(v * 1e9).toFixed(1)}ns`;
  if (a < 1e-3) return `${(v * 1e6).toFixed(2)}µs`;
  return `${(v * 1e3).toFixed(2)}ms`;
}

// Levenshtein-1 nearest match for "did you mean" hints.
function nearest(word: string, candidates: string[]): string | null {
  let best: string | null = null;
  let bestD = Infinity;
  for (const c of candidates) {
    const d = lev(word, c);
    if (d < bestD) { bestD = d; best = c; }
  }
  return bestD <= Math.max(2, Math.floor(word.length / 3)) ? best : null;
}

function lev(a: string, b: string): number {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) => [i, ...Array(n).fill(0)]);
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
    }
  }
  return dp[m][n];
}
