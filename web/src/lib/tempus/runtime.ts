import { Program, CellDecl, SyncDecl, ComposeDecl, Stmt, ParsedProgram, SimEvent } from './types';

// ── Build runtime from parsed AST ─────────────────────────────────────────────
export function buildRuntime(prog: Program): ParsedProgram {
  const cells   = new Map<string, CellDecl>();
  const syncs   = new Map<string, SyncDecl>();
  const whens   = new Map<string, string[]>();
  let compose: ComposeDecl | null = null;

  for (const d of prog.decls) {
    switch (d.kind) {
      case 'cell':    cells.set(d.name, d);        break;
      case 'sync':    syncs.set(d.name, d);        break;
      case 'compose': compose = d;                 break;
      case 'when':    whens.set(d.cell, stmtLabels(d.stmt)); break;
    }
  }
  if (cells.size === 0) throw new Error('No cell declarations found');

  return { cells, syncs, compose, whens };
}

function stmtLabels(stmt: Stmt): string[] {
  switch (stmt.kind) {
    case 'emit':  return [`emit ${stmt.name}`];
    case 'fire':  return [`fire ${stmt.name}(${stmt.args.join(', ')})`];
    case 'wait':  return [`wait ${stmt.duration}`];
    case 'block': return stmt.stmts.flatMap(stmtLabels);
  }
}

// ── Seeded PRNG (Mulberry32) ──────────────────────────────────────────────────
function mulberry32(seed: number) {
  let s = seed;
  return () => {
    s |= 0; s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// Box-Muller Gaussian sample
function gauss(r: () => number): number {
  const u = r(), v = r();
  return Math.sqrt(-2 * Math.log(u + 1e-12)) * Math.cos(2 * Math.PI * v);
}

// ── Simulator ─────────────────────────────────────────────────────────────────
export interface SimConfig {
  totalEvents: number;
  noiseSigma:  number;   // 0 = no noise, 1 = cell-width sigma
  seed:        number;
  batchSize:   number;
}

export function createSimulator(prog: ParsedProgram, cfg: SimConfig) {
  const rand      = mulberry32(cfg.seed);
  const cellList  = Array.from(prog.cells.values());
  const channels  = prog.compose?.channels ?? Array.from(prog.syncs.keys());
  if (channels.length === 0) throw new Error('No channels declared');

  // trajectory length = number of channels (n = d)
  const n    = channels.length;
  const freq = prog.syncs.get(channels[0])?.freq ?? 10e6;

  // cell weights proportional to width
  const widths    = cellList.map(c => c.hi - c.lo);
  const totalW    = widths.reduce((a, b) => a + b, 0);
  const cumWeight = widths.map((_, i) => widths.slice(0, i + 1).reduce((a, b) => a + b, 0) / totalW);

  let events:     SimEvent[] = [];
  let trajBuf:    { cell: string; dp: number; ch: string }[] = [];
  let trajId      = 0;
  let M           = 0;
  let chanIdx     = 0;

  function sampleCell(): CellDecl {
    const r = rand();
    return cellList[cumWeight.findIndex(w => r <= w)] ?? cellList[cellList.length - 1];
  }

  function classifyDp(dp: number): string {
    for (const c of cellList) if (dp >= c.lo && dp <= c.hi) return c.name;
    return 'anomaly';
  }

  function generateBatch(): SimEvent[] {
    const batch: SimEvent[] = [];

    for (let b = 0; b < cfg.batchSize; b++) {
      if (events.length >= cfg.totalEvents) break;

      M++;
      const ch        = channels[chanIdx % channels.length];
      chanIdx++;

      // ΔP: sample inside a randomly chosen cell, add noise
      const target    = sampleCell();
      const half      = (target.hi - target.lo) / 2;
      const mid       = target.lo + half;
      const noiseSd   = half * cfg.noiseSigma;
      const dp        = mid + half * (rand() * 2 - 1) + (noiseSd > 0 ? noiseSd * gauss(rand) : 0);
      const cellName  = classifyDp(dp);

      trajBuf.push({ cell: cellName, dp, ch });

      let actionFired: string | null = null;
      let phase: 'COMPILE' | 'EXECUTE' = 'COMPILE';

      if (trajBuf.length >= n) {
        // trajectory complete → dispatch action for the last cell hit
        phase = 'EXECUTE';
        const last = trajBuf[trajBuf.length - 1].cell;
        const acts = prog.whens.get(last);
        actionFired = acts ? acts[0] : null;
        trajBuf = [];
        trajId++;
      }

      batch.push({
        index:        events.length + batch.length,
        channel:      ch,
        dp,
        cell:         cellName,
        M,
        phase,
        trajectoryId: trajId,
        actionFired,
        time:         M / freq,
      });
    }

    events = events.concat(batch);
    return batch;
  }

  return {
    generateBatch,
    isDone:    () => events.length >= cfg.totalEvents,
    getEvents: () => events,
    getCells:  () => prog.cells,
    getChannels: () => channels,
    reset: () => { events = []; trajBuf = []; trajId = 0; M = 0; chanIdx = 0; },
  };
}
