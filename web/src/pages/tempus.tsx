import React, { useState, useEffect, useRef, useCallback } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { lex }           from '../lib/tempus/lexer';
import { parse }         from '../lib/tempus/parser';
import { buildRuntime, createSimulator } from '../lib/tempus/runtime';
import type { SimEvent, CellDecl } from '../lib/tempus/types';

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

// ── Colours ───────────────────────────────────────────────────────────────────
const PALETTE = ['#58E6D9','#f59e0b','#f97316','#ef4444','#60a5fa','#a78bfa','#34d399','#fb923c'];
const ANOMALY = '#4b5563';

function cellPalette(names: string[]): Map<string, string> {
  const m = new Map<string, string>();
  names.forEach((n, i) => m.set(n, PALETTE[i % PALETTE.length]));
  m.set('anomaly', ANOMALY);
  return m;
}

// ── Axis helpers ─────────────────────────────────────────────────────────────
function fmtSci(v: number): string {
  if (v === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = v / Math.pow(10, e);
  return `${m.toFixed(1)}e${e}`;
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  W: number, H: number,
  padL: number, padR: number, padT: number, padB: number
) {
  const iW = W - padL - padR;
  const iH = H - padT - padB;
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 0.5;
  for (let i = 1; i < 5; i++) {
    const x = padL + (iW * i) / 5;
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + iH); ctx.stroke();
  }
  for (let i = 1; i < 4; i++) {
    const y = padT + (iH * i) / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + iW, y); ctx.stroke();
  }
}

// ── Chart 1: ΔP scatter with cell bands ──────────────────────────────────────
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
  ctx.fillStyle = '#0a0a12';
  ctx.fillRect(0, 0, W, H);

  const PAD = { l: 46, r: 10, t: 10, b: 28 };
  const iW = W - PAD.l - PAD.r;
  const iH = H - PAD.t - PAD.b;

  const cellArr = Array.from(cells.values());
  const yMin = Math.min(...cellArr.map(c => c.lo));
  const yMax = Math.max(...cellArr.map(c => c.hi));
  const yRange = yMax - yMin || 1;

  const xS = (i: number) => PAD.l + (i / Math.max(events.length - 1, 1)) * iW;
  const yS = (v: number) => PAD.t + iH - ((v - yMin) / yRange) * iH;

  // cell band fills
  for (const [name, cell] of cells) {
    const col = colors.get(name) ?? '#888';
    const y1  = yS(cell.hi);
    const y2  = yS(cell.lo);
    ctx.fillStyle = col + '18';
    ctx.fillRect(PAD.l, y1, iW, y2 - y1);
    ctx.strokeStyle = col + '45';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(PAD.l, y1); ctx.lineTo(PAD.l + iW, y1); ctx.stroke();
    // band label
    ctx.fillStyle = col + '90';
    ctx.font = '8px Montserrat, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(name, PAD.l + 3, (y1 + y2) / 2 + 3);
  }

  drawGrid(ctx, W, H, PAD.l, PAD.r, PAD.t, PAD.b);

  // dots
  for (const ev of events) {
    const x   = xS(ev.index);
    const y   = yS(ev.dp);
    const col = colors.get(ev.cell) ?? ANOMALY;
    ctx.beginPath();
    ctx.arc(x, y, ev.actionFired ? 3 : 1.8, 0, Math.PI * 2);
    ctx.fillStyle = ev.actionFired ? col : col + 'aa';
    ctx.fill();
  }

  // axes labels
  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  ctx.font = '8px Montserrat, monospace';
  ctx.textAlign = 'right';
  ctx.fillText(fmtSci(yMax), PAD.l - 2, PAD.t + 6);
  ctx.fillText(fmtSci(yMin), PAD.l - 2, PAD.t + iH);
  ctx.textAlign = 'left';
  ctx.fillText('0', PAD.l + 2, PAD.t + iH + 10);
  ctx.textAlign = 'right';
  ctx.fillText(`${events.length}`, PAD.l + iW, PAD.t + iH + 10);

  // chart title
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '9px Montserrat, monospace';
  ctx.textAlign = 'left';
  ctx.fillText('ΔP  TIMELINE', PAD.l + 4, PAD.t + 9);
}

// ── Chart 2: cell hit frequency (bar) ────────────────────────────────────────
function drawCellBar(
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
  ctx.fillStyle = '#0a0a12';
  ctx.fillRect(0, 0, W, H);

  const PAD = { l: 10, r: 10, t: 18, b: 36 };
  const names = [...Array.from(cells.keys()), 'anomaly'];
  const counts = new Map<string, number>();
  names.forEach(n => counts.set(n, 0));
  events.forEach(e => counts.set(e.cell, (counts.get(e.cell) ?? 0) + 1));

  const maxCount = Math.max(...Array.from(counts.values()), 1);
  const iW = W - PAD.l - PAD.r;
  const iH = H - PAD.t - PAD.b;
  const bW  = iW / names.length;

  drawGrid(ctx, W, H, PAD.l, PAD.r, PAD.t, PAD.b);

  names.forEach((name, i) => {
    const count = counts.get(name) ?? 0;
    const col   = colors.get(name) ?? ANOMALY;
    const x     = PAD.l + i * bW;
    const bH    = (count / maxCount) * iH;
    const y     = PAD.t + iH - bH;

    ctx.fillStyle = col + '30';
    ctx.fillRect(x + 2, PAD.t, bW - 4, iH);
    ctx.fillStyle = col;
    ctx.fillRect(x + 2, y, bW - 4, bH);

    // label
    ctx.fillStyle = col;
    ctx.font = '8px Montserrat, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(name, x + bW / 2, H - PAD.b + 12);
    if (count > 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fillText(String(count), x + bW / 2, y - 3);
    }
  });

  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '9px Montserrat, monospace';
  ctx.textAlign = 'left';
  ctx.fillText('CELL  FREQUENCY', PAD.l + 4, PAD.t - 4);
}

// ── Chart 3: channel event distribution (horizontal bars) ────────────────────
function drawChannelBar(
  canvas:   HTMLCanvasElement | null,
  events:   SimEvent[],
  channels: string[]
) {
  if (!canvas) return;
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = canvas.offsetHeight;
  if (W === 0 || H === 0) return;

  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0a12';
  ctx.fillRect(0, 0, W, H);

  const PAD = { l: 70, r: 14, t: 18, b: 14 };
  const counts = new Map<string, number>();
  channels.forEach(c => counts.set(c, 0));
  events.forEach(e => counts.set(e.channel, (counts.get(e.channel) ?? 0) + 1));
  const maxC = Math.max(...Array.from(counts.values()), 1);
  const iW   = W - PAD.l - PAD.r;
  const iH   = H - PAD.t - PAD.b;
  const rowH = iH / Math.max(channels.length, 1);

  channels.forEach((ch, i) => {
    const count = counts.get(ch) ?? 0;
    const barW  = (count / maxC) * iW;
    const y     = PAD.t + i * rowH + rowH * 0.15;
    const bH    = rowH * 0.7;

    ctx.fillStyle = PALETTE[i % PALETTE.length] + '25';
    ctx.fillRect(PAD.l, y, iW, bH);
    ctx.fillStyle = PALETTE[i % PALETTE.length];
    ctx.fillRect(PAD.l, y, barW, bH);

    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = '9px Montserrat, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(ch, PAD.l - 4, y + bH * 0.72);
    ctx.textAlign = 'left';
    if (count > 0) ctx.fillText(String(count), PAD.l + barW + 4, y + bH * 0.72);
  });

  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '9px Montserrat, monospace';
  ctx.textAlign = 'left';
  ctx.fillText('CHANNEL  ACTIVITY', PAD.l + 4, PAD.t - 4);
}

// ── Chart 4: action dispatch counts (bar) ────────────────────────────────────
function drawActionBar(
  canvas: HTMLCanvasElement | null,
  events: SimEvent[],
  colors: Map<string, string>
) {
  if (!canvas) return;
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = canvas.offsetHeight;
  if (W === 0 || H === 0) return;

  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0a12';
  ctx.fillRect(0, 0, W, H);

  const PAD = { l: 10, r: 10, t: 18, b: 44 };

  const counts = new Map<string, number>();
  events.forEach(e => {
    if (e.actionFired) counts.set(e.actionFired, (counts.get(e.actionFired) ?? 0) + 1);
  });
  const entries = Array.from(counts.entries());
  if (entries.length === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.font = '9px Montserrat, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('no dispatches yet', W / 2, H / 2);
    ctx.textAlign = 'left';
    ctx.fillText('ACTION  DISPATCH', PAD.l + 4, PAD.t - 4);
    return;
  }

  const maxC = Math.max(...entries.map(([, v]) => v));
  const iW   = W - PAD.l - PAD.r;
  const iH   = H - PAD.t - PAD.b;
  const bW   = iW / entries.length;

  drawGrid(ctx, W, H, PAD.l, PAD.r, PAD.t, PAD.b);

  entries.forEach(([action, count], i) => {
    const col = PALETTE[i % PALETTE.length];
    const x   = PAD.l + i * bW;
    const bH  = (count / maxC) * iH;
    const y   = PAD.t + iH - bH;

    ctx.fillStyle = col + '28';
    ctx.fillRect(x + 2, PAD.t, bW - 4, iH);
    ctx.fillStyle = col;
    ctx.fillRect(x + 2, y, bW - 4, bH);

    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '7px Montserrat, monospace';
    ctx.textAlign = 'center';
    // truncate long action strings
    const label = action.length > 18 ? action.slice(0, 16) + '…' : action;
    ctx.save();
    ctx.translate(x + bW / 2, H - PAD.b + 6);
    ctx.rotate(-Math.PI / 5);
    ctx.fillText(label, 0, 0);
    ctx.restore();

    if (count > 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.font = '8px Montserrat, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(String(count), x + bW / 2, y - 3);
    }
  });

  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '9px Montserrat, monospace';
  ctx.textAlign = 'left';
  ctx.fillText('ACTION  DISPATCH', PAD.l + 4, PAD.t - 4);
}

// ── Page ─────────────────────────────────────────────────────────────────────
interface Stats { total: number; dispatched: number; anomalies: number; done: boolean; }

function TempusSandbox() {
  const [code,    setCode]    = useState(DEFAULT);
  const [nEvents, setNEvents] = useState(600);
  const [noise,   setNoise]   = useState(0.35);
  const [speed,   setSpeed]   = useState(200);   // events / second
  const [error,   setError]   = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [stats,   setStats]   = useState<Stats>({ total: 0, dispatched: 0, anomalies: 0, done: false });

  const simRef      = useRef<ReturnType<typeof createSimulator> | null>(null);
  const colorsRef   = useRef<Map<string, string>>(new Map());
  const channelsRef = useRef<string[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const dpRef   = useRef<HTMLCanvasElement>(null);
  const cellRef = useRef<HTMLCanvasElement>(null);
  const chanRef = useRef<HTMLCanvasElement>(null);
  const actRef  = useRef<HTMLCanvasElement>(null);

  const redrawAll = useCallback(() => {
    if (!simRef.current) return;
    const events = simRef.current.getEvents();
    const cells  = simRef.current.getCells();
    drawDpScatter  (dpRef.current,   events, cells, colorsRef.current);
    drawCellBar    (cellRef.current,  events, cells, colorsRef.current);
    drawChannelBar (chanRef.current,  events, channelsRef.current);
    drawActionBar  (actRef.current,   events, colorsRef.current);
    setStats({
      total:      events.length,
      dispatched: events.filter(e => e.actionFired !== null).length,
      anomalies:  events.filter(e => e.cell === 'anomaly').length,
      done:       simRef.current.isDone(),
    });
  }, []);

  const handleRun = useCallback(() => {
    try {
      setError(null);
      const prog = buildRuntime(parse(lex(code)));
      colorsRef.current   = cellPalette(Array.from(prog.cells.keys()));
      channelsRef.current = prog.compose?.channels ?? Array.from(prog.syncs.keys());
      const batchSize = Math.max(1, Math.floor(speed / 20));
      simRef.current = createSimulator(prog, { totalEvents: nEvents, noiseSigma: noise, seed: 42, batchSize });
      setRunning(true);
    } catch (e: any) {
      setError(String(e.message));
    }
  }, [code, nEvents, noise, speed]);

  const handleStop = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setRunning(false);
  }, []);

  const handleReset = useCallback(() => {
    handleStop();
    simRef.current?.reset();
    redrawAll();
    setStats({ total: 0, dispatched: 0, anomalies: 0, done: false });
  }, [handleStop, redrawAll]);

  // simulation loop
  useEffect(() => {
    if (!running) return;
    intervalRef.current = setInterval(() => {
      if (!simRef.current || simRef.current.isDone()) {
        setRunning(false);
        return;
      }
      simRef.current.generateBatch();
      redrawAll();
    }, 50);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running, redrawAll]);

  // resize observer — redraw when panel resizes
  useEffect(() => {
    const canvases = [dpRef, cellRef, chanRef, actRef].map(r => r.current).filter(Boolean);
    if (!canvases.length) return;
    const ro = new ResizeObserver(() => redrawAll());
    canvases.forEach(c => ro.observe(c!));
    return () => ro.disconnect();
  }, [redrawAll]);

  const pct = nEvents > 0 ? Math.round((stats.total / nEvents) * 100) : 0;

  // ── JSX ────────────────────────────────────────────────────────────────────
  return (
    <div style={{ background:'#030705', color:'#e2e8f0', minHeight:'100vh',
                  fontFamily:'Montserrat, monospace', display:'flex', flexDirection:'column' }}>
      <Head><title>Tempus Sandbox</title></Head>

      {/* ── header ── */}
      <header style={{ height:56, borderBottom:'1px solid rgba(255,255,255,0.07)',
                       display:'flex', alignItems:'center', padding:'0 24px', gap:20, flexShrink:0 }}>
        <Link href="/" style={{ color:'rgba(255,255,255,0.4)', fontSize:12,
                                textDecoration:'none', letterSpacing:2 }}>← BACK</Link>
        <span style={{ color:'rgba(255,255,255,0.9)', fontSize:12, letterSpacing:4, fontWeight:600 }}>
          TEMPUS  SANDBOX
        </span>
        <div style={{ marginLeft:'auto', display:'flex', gap:8 }}>
          <button onClick={handleRun} disabled={running}
            style={{ padding:'5px 18px', fontSize:11, letterSpacing:2, cursor: running ? 'not-allowed' : 'pointer',
                     background: running ? '#1a2a1a' : '#58E6D9', color: running ? '#445' : '#020f0d',
                     border:'none', fontFamily:'inherit', fontWeight:700 }}>
            RUN
          </button>
          <button onClick={handleStop} disabled={!running}
            style={{ padding:'5px 18px', fontSize:11, letterSpacing:2, cursor: !running ? 'not-allowed' : 'pointer',
                     background:'transparent', color: !running ? 'rgba(255,255,255,0.2)' : '#ef4444',
                     border:`1px solid ${!running ? 'rgba(255,255,255,0.1)' : '#ef4444'}`, fontFamily:'inherit' }}>
            STOP
          </button>
          <button onClick={handleReset}
            style={{ padding:'5px 18px', fontSize:11, letterSpacing:2, cursor:'pointer',
                     background:'transparent', color:'rgba(255,255,255,0.35)',
                     border:'1px solid rgba(255,255,255,0.12)', fontFamily:'inherit' }}>
            RESET
          </button>
        </div>
      </header>

      {/* ── body: editor | charts ── */}
      <div style={{ display:'flex', flex:1, overflow:'hidden', minHeight:0 }}>

        {/* ── LEFT: editor + controls ── */}
        <div style={{ width:420, flexShrink:0, borderRight:'1px solid rgba(255,255,255,0.07)',
                      display:'flex', flexDirection:'column', overflow:'hidden' }}>

          {/* code editor */}
          <textarea
            value={code}
            onChange={e => setCode(e.target.value)}
            spellCheck={false}
            style={{ flex:1, background:'#070c0a', color:'#c8faf5', fontFamily:'monospace',
                     fontSize:12, lineHeight:1.6, padding:'16px', border:'none', resize:'none',
                     outline:'none', minHeight:0 }}
          />

          {/* error */}
          {error && (
            <div style={{ background:'#2a0a0a', color:'#ef4444', fontSize:11,
                          padding:'8px 14px', borderTop:'1px solid #4a1515' }}>
              {error}
            </div>
          )}

          {/* sim controls */}
          <div style={{ padding:'14px 16px', borderTop:'1px solid rgba(255,255,255,0.07)',
                        display:'flex', flexDirection:'column', gap:10 }}>

            <div style={{ display:'flex', gap:12, alignItems:'center' }}>
              <label style={{ fontSize:10, letterSpacing:2, color:'rgba(255,255,255,0.4)', width:80 }}>
                EVENTS
              </label>
              <input type="number" value={nEvents} min={50} max={5000} step={50}
                onChange={e => setNEvents(+e.target.value)}
                style={{ width:80, background:'#0d1a14', border:'1px solid rgba(255,255,255,0.1)',
                         color:'#c8faf5', fontFamily:'monospace', fontSize:12, padding:'3px 6px' }}
              />
            </div>

            <div style={{ display:'flex', gap:12, alignItems:'center' }}>
              <label style={{ fontSize:10, letterSpacing:2, color:'rgba(255,255,255,0.4)', width:80 }}>
                NOISE  σ
              </label>
              <input type="range" min={0} max={1} step={0.05} value={noise}
                onChange={e => setNoise(+e.target.value)}
                style={{ flex:1, accentColor:'#58E6D9' }}
              />
              <span style={{ fontSize:11, color:'#58E6D9', width:32 }}>{noise.toFixed(2)}</span>
            </div>

            <div style={{ display:'flex', gap:12, alignItems:'center' }}>
              <label style={{ fontSize:10, letterSpacing:2, color:'rgba(255,255,255,0.4)', width:80 }}>
                SPEED
              </label>
              <input type="range" min={20} max={1000} step={20} value={speed}
                onChange={e => setSpeed(+e.target.value)}
                style={{ flex:1, accentColor:'#58E6D9' }}
              />
              <span style={{ fontSize:11, color:'#58E6D9', width:54 }}>{speed}/s</span>
            </div>

            {/* progress + stats */}
            <div style={{ marginTop:4 }}>
              <div style={{ background:'rgba(255,255,255,0.06)', height:3, borderRadius:2 }}>
                <div style={{ background:'#58E6D9', height:'100%', borderRadius:2,
                              width:`${pct}%`, transition:'width 0.15s' }} />
              </div>
              <div style={{ display:'flex', justifyContent:'space-between', marginTop:8 }}>
                {[
                  ['EVENTS',    `${stats.total} / ${nEvents}`],
                  ['DISPATCH',  String(stats.dispatched)],
                  ['ANOMALY',   String(stats.anomalies)],
                  ['STATUS',    stats.done ? 'DONE' : running ? 'RUN' : 'IDLE'],
                ].map(([k, v]) => (
                  <div key={k} style={{ textAlign:'center' }}>
                    <div style={{ fontSize:8, letterSpacing:2, color:'rgba(255,255,255,0.3)' }}>{k}</div>
                    <div style={{ fontSize:11, color: k === 'STATUS' && running ? '#58E6D9'
                                                    : k === 'ANOMALY' && stats.anomalies > 0 ? '#ef4444'
                                                    : 'rgba(255,255,255,0.7)' }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* ── RIGHT: 2×2 chart grid ── */}
        <div style={{ flex:1, display:'grid', gridTemplateColumns:'1fr 1fr',
                      gridTemplateRows:'1fr 1fr', gap:1, background:'rgba(255,255,255,0.05)',
                      minWidth:0 }}>
          {[
            { ref: dpRef,   label: 'ΔP TIMELINE'     },
            { ref: cellRef, label: 'CELL FREQUENCY'  },
            { ref: chanRef, label: 'CHANNEL ACTIVITY'},
            { ref: actRef,  label: 'ACTION DISPATCH' },
          ].map(({ ref, label }) => (
            <div key={label} style={{ background:'#030705', position:'relative', overflow:'hidden' }}>
              <canvas ref={ref as React.RefObject<HTMLCanvasElement>}
                      style={{ width:'100%', height:'100%', display:'block' }} />
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}

TempusSandbox.getLayout = (page: React.ReactElement) => page;
export default TempusSandbox;
