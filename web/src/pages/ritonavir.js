import Head from "next/head";
import Link from "next/link";
import { useState, useMemo, useRef, useEffect, useCallback } from "react";

// ── Seeded PRNG (Mulberry32) for deterministic synthetic data ─────────────────
function mulberry32(seed) {
  let s = seed | 0;
  return () => {
    s += 0x9e3779b9; let t = s ^ (s >>> 11);
    t ^= t << 7 & 0x9d2c5680; t ^= t << 15 & 0xefc60000; t ^= t >>> 18;
    return (t >>> 0) / 0x100000000;
  };
}

// ── 120-batch synthetic manufacturing dataset ─────────────────────────────────
// cooling_rate → form_II_pct is the root causal link (Abbott 1998 crisis).
function generateBatches() {
  const rand = mulberry32(0xdeadbeef);
  const randn = () => {
    const u = rand() + 1e-10;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  const out = [];
  const t0 = new Date(2021, 0, 15).getTime();
  for (let i = 0; i < 120; i++) {
    const date      = new Date(t0 + i * 9 * 86400000);
    const cooling   = +(Math.max(0.05, 0.82 + randn() * 0.50)).toFixed(2);
    const seed_time = Math.max(0, Math.round(65 + randn() * 28));
    const rxn_temp  = +(Math.max(0, 15 + randn() * 8)).toFixed(1);
    const f2base    = Math.max(0, (cooling - 0.42) * 16 + (seed_time - 60) * 0.08);
    const form_II   = +(Math.min(46, Math.max(0, f2base + randn() * 2.2))).toFixed(1);
    const yld       = +(Math.min(97, Math.max(60, 88 - rxn_temp * 0.25 - form_II * 0.15 + randn() * 3.5))).toFixed(1);
    const purity    = +(Math.min(99.9, Math.max(78, 97.5 - form_II * 0.35 - Math.abs(randn()) * 1.2))).toFixed(2);
    out.push({
      id: i + 1, date,
      month: new Date(date.getFullYear(), date.getMonth(), 1),
      cooling_rate: cooling,
      seed_time,
      rxn_temp,
      form_II_pct: form_II,
      yield_pct: yld,
      purity,
      passed: form_II < 5 && purity > 95,
    });
  }
  return out;
}
const BATCHES = generateBatches();

// ── Filter state ──────────────────────────────────────────────────────────────
function applyFilters(data, filters) {
  return data.filter(d =>
    Object.entries(filters).every(([k, r]) => !r || (d[k] >= r[0] && d[k] <= r[1]))
  );
}

// ── Axis metadata ─────────────────────────────────────────────────────────────
const AXIS = {
  cooling_rate: { domain: [0, 2.5],  label: "Cooling rate (°C/min)", spec: 0.5  },
  form_II_pct:  { domain: [0, 48],   label: "Form II (%)",           spec: 5    },
  rxn_temp:     { domain: [0, 36],   label: "Reaction temp (°C)",    spec: 15   },
  yield_pct:    { domain: [58, 100], label: "Yield (%)"                         },
  purity:       { domain: [78, 100], label: "Purity (%)",            spec: 95   },
};

// ── Circuit symbols (SVG, centered at 0,0) ────────────────────────────────────
const SYMBOLS = {
  battery: ({ c }) => (
    <>
      <line x1="-14" y1="-5" x2="14" y2="-5" strokeWidth="3" />
      <line x1="-9"  y1="5"  x2="9"  y2="5"  strokeWidth="1.5" />
      <text x="18" y="-4" fontSize="7" fill={c} opacity="0.55" dominantBaseline="middle">+</text>
    </>
  ),
  inductor: ({ c }) => (
    <>
      {[-18, -6, 6, 18].map((x, i) => (
        <path key={i} d={`M${x - 6},0 a6,6 0 0,1 12,0`} strokeWidth="1.5" fill="none" />
      ))}
    </>
  ),
  capacitor: ({ c }) => (
    <>
      <line x1="-6" y1="-14" x2="-6" y2="14" strokeWidth="2.5" />
      <line x1="6"  y1="-14" x2="6"  y2="14" strokeWidth="2.5" />
    </>
  ),
  zener: ({ c }) => (
    <>
      <polygon points="-12,-10 -12,10 8,0" fill={`${c}2a`} strokeWidth="1.5" />
      <line x1="8"  y1="-14" x2="8"  y2="14" strokeWidth="2"   />
      <line x1="8"  y1="-14" x2="16" y2="-8" strokeWidth="1.5" />
      <line x1="8"  y1="14"  x2="0"  y2="8"  strokeWidth="1.5" />
    </>
  ),
  switch: ({ c }) => (
    <>
      <circle cx="0" cy="-10" r="2.5" strokeWidth="1.5" fill="none" />
      <circle cx="0" cy="10"  r="2.5" strokeWidth="1.5" fill="none" />
      <line x1="0" y1="-7.5" x2="14" y2="-20" strokeWidth="1.5" />
    </>
  ),
  resistor: ({ c }) => (
    <rect x="-20" y="-9" width="40" height="18" rx="2" strokeWidth="1.5" fill="none" />
  ),
  oscilloscope: ({ c }) => (
    <>
      <circle r="18" strokeWidth="1.5" fill="none" />
      <path d="M-12,0 C-8,-9 -4,9 0,0 C4,-9 8,9 12,0" strokeWidth="1.5" fill="none" />
    </>
  ),
};

// ── Node definitions ──────────────────────────────────────────────────────────
const NODES = [
  {
    id: "raw", label: "Raw Materials", sub: "VAL · THZ · REAGENTS",
    x: 90, y: 190, symbol: "battery", orient: "v", color: "#58E6D9",
    desc: "Boc-L-valinol, 2-amino-4-methylthiazole, peptide coupling agents (EDC·HCl / HOBt / DIPEA). Stored at −20°C under inert atmosphere.",
    specs: [
      ["Key fragment",    "N-Boc-L-valinol"],
      ["Thiazole source", "2-Amino-4-methylthiazole"],
      ["Coupling agent",  "EDC·HCl / HOBt / DIPEA"],
      ["Solvent / T",     "Dry DMF, −20°C"],
    ],
    charts: [],
  },
  {
    id: "amide", label: "Amide Formation", sub: "CIS/TRANS CONTROL",
    x: 275, y: 100, symbol: "inductor", orient: "h", color: "#B63E96",
    desc: "Peptide coupling at low temperature. This bond is the molecular origin of Form I vs II — the cis amide (Form I) vs trans amide (Form II). T > 15°C leaks toward trans.",
    specs: [
      ["Target temperature", "0–5°C"],
      ["Selectivity",        "> 98% cis at T < 5°C"],
      ["Duration",           "4–6 h"],
      ["Monitoring",         "HPLC IPC (cis/trans ratio)"],
    ],
    charts: ["scatter:rxn_temp:form_II_pct"],
  },
  {
    id: "workup", label: "Aqueous Workup", sub: "EXTRACTION · pH 7.4",
    x: 468, y: 100, symbol: "capacitor", orient: "h", color: "#58E6D9",
    desc: "pH-controlled extraction removes HOBt byproduct and coupling reagents. Three-stage EtOAc wash. Purity > 95% is required before crystallization.",
    specs: [
      ["Base wash",    "5% NaHCO₃, pH 7.4"],
      ["Extraction",   "EtOAc × 3"],
      ["Drying agent", "Na₂SO₄ / MgSO₄"],
      ["IPC spec",     "Area% > 95% HPLC"],
    ],
    charts: ["histogram:purity"],
  },
  {
    id: "cryst", label: "Crystallization", sub: "POLYMORPH CONTROL",
    x: 678, y: 100, symbol: "zener", orient: "h", color: "#f59e0b", critical: true,
    desc: "CRITICAL. Slow cooling from EtOH/H₂O selects Form I (cis, soluble). Too-fast cooling or late seeding nucleates Form II — 400× less soluble. Root cause of the 1998 Abbott Norvir crisis.",
    specs: [
      ["Solvent",      "EtOH / H₂O 9:1 v/v"],
      ["Cooling spec", "< 0.5 °C/min  ← CRITICAL"],
      ["Seed addition","45°C, before nucleation"],
      ["Seed type",    "Form I authenticated crystals"],
    ],
    charts: ["scatter:cooling_rate:form_II_pct", "histogram:form_II_pct"],
  },
  {
    id: "filter", label: "Filtration", sub: "NUTSCHE · 5 μm PP",
    x: 850, y: 190, symbol: "switch", orient: "v", color: "#58E6D9",
    desc: "Pressure filtration. Form I crystal habit (elongated needles) gives good cake permeability. Cold ethanol wash removes occluded mother liquor.",
    specs: [
      ["Filter medium",  "PP 5 μm sintered disc"],
      ["Wash",           "Cold EtOH × 2 vol"],
      ["LOD target",     "< 0.5% w/w"],
      ["Crystal habit",  "Form I: elongated needles"],
    ],
    charts: [],
  },
  {
    id: "dry", label: "Drying", sub: "VAC TRAY 40°C · 20 mbar",
    x: 596, y: 280, symbol: "resistor", orient: "h", color: "#58E6D9",
    desc: "Vacuum tray drying. Over-drying (T > 60°C) fractures crystals; under-drying leaves EtOH above ICH Q3C limit.",
    specs: [
      ["Conditions",   "40°C, 20 mbar vacuum"],
      ["Duration",     "12–24 h"],
      ["Moisture",     "NMT 0.5% w/w KF"],
      ["Residual EtOH","NMT 5000 ppm ICH Q3C"],
    ],
    charts: [],
  },
  {
    id: "qc", label: "QC Analytics", sub: "FTIR · DSC · XRPD",
    x: 345, y: 280, symbol: "oscilloscope", orient: "h", color: "#B63E96",
    isAnalytics: true,
    desc: "FTIR identifies polymorph by Amide I band. Form I at 1695 cm⁻¹, Form II at 1668 cm⁻¹. DSC and XRPD as orthogonal confirmation. Failing batches trace to Crystallization.",
    specs: [
      ["FTIR Form I",  "1695 cm⁻¹ Amide I"],
      ["FTIR Form II", "1668 cm⁻¹ Amide I"],
      ["DSC Form I",   "m.p. 121°C, ΔH = 47 kJ/mol"],
      ["Release spec", "Form II < 5%, Purity > 95%"],
    ],
    charts: ["scatter:yield_pct:purity", "histogram:form_II_pct"],
  },
];

// ── Canvas drawing helpers ────────────────────────────────────────────────────
const PAD = { l: 48, r: 16, t: 18, b: 44 };

function lerp(v, d0, d1, r0, r1) {
  return r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

function canvasToData(px, d0, d1, r0, r1) {
  return d0 + ((px - r0) / (r1 - r0)) * (d1 - d0);
}

function drawFrame(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#030a05";
  ctx.fillRect(0, 0, w, h);
  const pw = w - PAD.l - PAD.r, ph = h - PAD.t - PAD.b;
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const x = PAD.l + (i / 4) * pw;
    ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + ph); ctx.stroke();
    const y = PAD.t + (i / 4) * ph;
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + pw, y); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, PAD.t + ph + 1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(PAD.l - 1, PAD.t + ph); ctx.lineTo(PAD.l + pw, PAD.t + ph); ctx.stroke();
  return { pw, ph };
}

function drawTicksX(ctx, w, h, dom, ticks) {
  const { pw, ph } = { pw: w - PAD.l - PAD.r, ph: h - PAD.t - PAD.b };
  ctx.fillStyle = "rgba(255,255,255,0.32)";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  ticks.forEach(v => {
    const x = PAD.l + lerp(v, dom[0], dom[1], 0, pw);
    ctx.fillText(String(v), x, PAD.t + ph + 16);
  });
}

function drawTicksY(ctx, w, h, dom, ticks) {
  const { pw, ph } = { pw: w - PAD.l - PAD.r, ph: h - PAD.t - PAD.b };
  ctx.fillStyle = "rgba(255,255,255,0.32)";
  ctx.font = "9px monospace";
  ctx.textAlign = "right";
  ticks.forEach(v => {
    const y = PAD.t + ph - lerp(v, dom[0], dom[1], 0, ph);
    ctx.fillText(String(v), PAD.l - 5, y + 3);
  });
}

function drawSpecLine(ctx, w, h, axis, dom, isX) {
  if (axis?.spec == null) return;
  const pw = w - PAD.l - PAD.r, ph = h - PAD.t - PAD.b;
  ctx.save();
  ctx.strokeStyle = "rgba(245,158,11,0.45)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  if (isX) {
    const x = PAD.l + lerp(axis.spec, dom[0], dom[1], 0, pw);
    ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + ph);
  } else {
    const y = PAD.t + ph - lerp(axis.spec, dom[0], dom[1], 0, ph);
    ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + pw, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

// ── ScatterChart ──────────────────────────────────────────────────────────────
function ScatterChart({ data, xKey, yKey, onBrush, width = 420, height = 230 }) {
  const canvasRef = useRef(null);
  const dragRef   = useRef({ active: false, x0: 0, y0: 0, x1: 0, y1: 0 });
  const [sel, setSel] = useState(null); // {x1,y1,x2,y2} canvas coords

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const xDom = useMemo(() => AXIS[xKey]?.domain ?? [0, 1], [xKey]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const yDom = useMemo(() => AXIS[yKey]?.domain ?? [0, 1], [yKey]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    const { pw, ph } = drawFrame(ctx, w, h);

    drawSpecLine(ctx, w, h, AXIS[xKey], xDom, true);
    drawSpecLine(ctx, w, h, AXIS[yKey], yDom, false);

    data.forEach(d => {
      const x = PAD.l + lerp(d[xKey], xDom[0], xDom[1], 0, pw);
      const y = PAD.t + ph - lerp(d[yKey], yDom[0], yDom[1], 0, ph);
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.fillStyle = d.passed ? "rgba(34,197,94,0.72)" : "rgba(239,68,68,0.72)";
      ctx.fill();
    });

    if (sel) {
      ctx.fillStyle = "rgba(88,230,217,0.07)";
      ctx.strokeStyle = "rgba(88,230,217,0.55)";
      ctx.lineWidth = 1;
      const bx = Math.min(sel.x1, sel.x2), by = Math.min(sel.y1, sel.y2);
      const bw = Math.abs(sel.x2 - sel.x1), bh = Math.abs(sel.y2 - sel.y1);
      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeRect(bx, by, bw, bh);
    }

    const xTicks = 5, yTicks = 5;
    const xStep = (xDom[1] - xDom[0]) / xTicks;
    const yStep = (yDom[1] - yDom[0]) / yTicks;
    drawTicksX(ctx, w, h, xDom, Array.from({length: xTicks + 1}, (_, i) => +(xDom[0] + i * xStep).toFixed(1)));
    drawTicksY(ctx, w, h, yDom, Array.from({length: yTicks + 1}, (_, i) => +(yDom[0] + i * yStep).toFixed(0)));

    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText(AXIS[xKey]?.label ?? xKey, PAD.l + pw / 2, h - 6);

    ctx.save();
    ctx.translate(12, PAD.t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText(AXIS[yKey]?.label ?? yKey, 0, 0);
    ctx.restore();
  }, [data, sel, xKey, yKey, xDom, yDom]);

  const getPos = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const onDown = (e) => {
    const p = getPos(e);
    dragRef.current = { active: true, x0: p.x, y0: p.y, x1: p.x, y1: p.y };
    setSel({ x1: p.x, y1: p.y, x2: p.x, y2: p.y });
  };

  const onMove = (e) => {
    if (!dragRef.current.active) return;
    const p = getPos(e);
    dragRef.current.x1 = p.x; dragRef.current.y1 = p.y;
    setSel({ x1: dragRef.current.x0, y1: dragRef.current.y0, x2: p.x, y2: p.y });
  };

  const onUp = () => {
    if (!dragRef.current.active) return;
    dragRef.current.active = false;
    const { x0, y0, x1, y1 } = dragRef.current;
    const pw = width - PAD.l - PAD.r, ph = height - PAD.t - PAD.b;
    const dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0);
    if (dx > 8 || dy > 8) {
      const xa = [Math.min(x0, x1), Math.max(x0, x1)].map(px =>
        Math.max(xDom[0], Math.min(xDom[1], canvasToData(px, xDom[0], xDom[1], PAD.l, PAD.l + pw)))
      );
      const ya = [Math.min(y0, y1), Math.max(y0, y1)].map(py =>
        Math.max(yDom[0], Math.min(yDom[1], canvasToData(py, yDom[0], yDom[1], PAD.t + ph, PAD.t)))
      ).reverse();
      onBrush({ [xKey]: xa, [yKey]: ya });
    } else {
      onBrush(null);
    }
    setSel(null);
  };

  return (
    <canvas ref={canvasRef} width={width} height={height}
            style={{ cursor: "crosshair", display: "block", borderRadius: 4 }}
            onMouseDown={onDown} onMouseMove={onMove}
            onMouseUp={onUp} onMouseLeave={onUp} />
  );
}

// ── HistogramChart ────────────────────────────────────────────────────────────
function HistogramChart({ data, xKey, onBrush, width = 420, height = 170 }) {
  const canvasRef  = useRef(null);
  const [selBin, setSelBin] = useState(null);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const cfg  = useMemo(() => AXIS[xKey] ?? { domain: [0, 100], label: xKey }, [xKey]);
  const dom  = cfg.domain;
  const nBin = 8;
  const binW = (dom[1] - dom[0]) / nBin;

  const bins = useMemo(() => {
    const b = Array.from({ length: nBin }, (_, i) => ({
      lo: dom[0] + i * binW,
      hi: dom[0] + (i + 1) * binW,
      pass: 0,
      fail: 0,
    }));
    data.forEach(d => {
      const idx = Math.min(nBin - 1, Math.floor((d[xKey] - dom[0]) / binW));
      if (idx >= 0) { d.passed ? b[idx].pass++ : b[idx].fail++; }
    });
    return b;
  }, [data, xKey, dom, binW, nBin]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    const { pw, ph } = drawFrame(ctx, w, h);
    const maxCount = Math.max(1, ...bins.map(b => b.pass + b.fail));
    const gap = 2;
    const bw  = pw / nBin - gap;

    drawSpecLine(ctx, w, h, cfg, dom, true);

    bins.forEach((b, i) => {
      const x = PAD.l + (i / nBin) * pw + gap / 2;
      const total = b.pass + b.fail;
      const totalH = (total / maxCount) * ph;
      const failH  = (b.fail  / maxCount) * ph;
      const passH  = (b.pass  / maxCount) * ph;

      const isSelected = selBin === i;

      if (b.pass > 0) {
        ctx.fillStyle = isSelected ? "rgba(34,197,94,0.85)" : "rgba(34,197,94,0.55)";
        ctx.fillRect(x, PAD.t + ph - passH, bw, passH);
      }
      if (b.fail > 0) {
        ctx.fillStyle = isSelected ? "rgba(239,68,68,0.85)" : "rgba(239,68,68,0.55)";
        ctx.fillRect(x, PAD.t + ph - totalH, bw, failH);
      }

      if (total > 0) {
        ctx.fillStyle = "rgba(255,255,255,0.4)";
        ctx.font = "8px monospace";
        ctx.textAlign = "center";
        ctx.fillText(total, x + bw / 2, PAD.t + ph - totalH - 4);
      }
    });

    const step = (dom[1] - dom[0]) / 4;
    drawTicksX(ctx, w, h, dom, Array.from({length: 5}, (_, i) => +(dom[0] + i * step).toFixed(0)));

    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText(cfg.label, PAD.l + pw / 2, h - 6);
  }, [bins, selBin, cfg, dom, nBin]);

  const onClick = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    const px = e.clientX - r.left;
    const pw = width - PAD.l - PAD.r;
    const idx = Math.floor(((px - PAD.l) / pw) * nBin);
    if (idx < 0 || idx >= nBin) { setSelBin(null); onBrush(null); return; }
    if (selBin === idx) { setSelBin(null); onBrush(null); }
    else {
      setSelBin(idx);
      onBrush({ [xKey]: [bins[idx].lo, bins[idx].hi] });
    }
  };

  return (
    <canvas ref={canvasRef} width={width} height={height}
            style={{ cursor: "pointer", display: "block", borderRadius: 4 }}
            onClick={onClick} />
  );
}

// ── Node modal ────────────────────────────────────────────────────────────────
function NodeModal({ node, data, filters, onBrush, onClose }) {
  const parseChart = (spec) => {
    const [type, ...keys] = spec.split(":");
    return { type, keys };
  };

  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 50,
        background: "rgba(0,0,0,0.72)", display: "flex",
        alignItems: "center", justifyContent: "center",
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: "#080c0a",
          border: `1px solid ${node.color}44`,
          borderRadius: 8,
          maxWidth: 900, width: "calc(100vw - 32px)",
          maxHeight: "90vh", overflow: "auto",
          boxShadow: `0 0 40px ${node.color}22`,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal header */}
        <div
          className="flex items-center justify-between px-7 py-5"
          style={{ borderBottom: `1px solid rgba(255,255,255,0.07)` }}
        >
          <div>
            <h2 className="text-sm tracking-widest uppercase font-medium"
                style={{ color: node.color }}>{node.label}</h2>
            <p className="text-xs mt-0.5 opacity-35 tracking-wider">{node.sub}</p>
          </div>
          <button onClick={onClose}
                  className="text-xs tracking-widest uppercase opacity-40 hover:opacity-80 transition-opacity">
            Close ×
          </button>
        </div>

        {/* Modal body */}
        <div className="flex gap-0" style={{ minHeight: 320 }}>
          {/* Left: info */}
          <div className="px-7 py-6 flex flex-col gap-5"
               style={{ width: 300, flexShrink: 0, borderRight: "1px solid rgba(255,255,255,0.06)" }}>
            <p className="text-xs opacity-55 leading-relaxed">{node.desc}</p>
            <div>
              <p className="text-xs tracking-widest uppercase opacity-35 mb-3">Process Parameters</p>
              <table className="w-full" style={{ fontSize: "0.62rem", borderCollapse: "collapse" }}>
                <tbody>
                  {node.specs.map(([k, v]) => (
                    <tr key={k} style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                      <td className="py-1.5 pr-3 opacity-45">{k}</td>
                      <td className="py-1.5" style={{ color: node.color, opacity: 0.85 }}>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {node.isAnalytics && (
              <div className="mt-auto">
                <Link href="/polymorphism"
                      className="block text-center text-xs tracking-widest uppercase py-2 rounded transition-opacity"
                      style={{ border: "1px solid rgba(182,62,150,0.4)", color: "#B63E96", opacity: 0.8 }}>
                  Open Spectral Hologram →
                </Link>
              </div>
            )}

            {/* Batch stats for this filter */}
            <div className="rounded p-3" style={{
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.06)",
              fontSize: "0.6rem",
            }}>
              <div className="flex justify-between mb-1.5">
                <span className="opacity-40">Batches in filter</span>
                <span style={{ color: node.color }}>{data.length} / {BATCHES.length}</span>
              </div>
              <div className="flex justify-between mb-1.5">
                <span className="opacity-40">Passing</span>
                <span style={{ color: "#22c55e" }}>{data.filter(d => d.passed).length}</span>
              </div>
              <div className="flex justify-between">
                <span className="opacity-40">Failing (Form II &gt; 5%)</span>
                <span style={{ color: "#ef4444" }}>{data.filter(d => !d.passed).length}</span>
              </div>
            </div>
          </div>

          {/* Right: charts */}
          <div className="flex-1 px-6 py-6 flex flex-col gap-5">
            {node.charts.length === 0 ? (
              <div className="flex-1 flex items-center justify-center opacity-25 text-xs tracking-widest uppercase">
                No process data for this node
              </div>
            ) : (
              node.charts.map((spec) => {
                const { type, keys } = parseChart(spec);
                if (type === "scatter") {
                  return (
                    <div key={spec}>
                      <p className="text-xs tracking-widest uppercase opacity-35 mb-2">
                        {AXIS[keys[0]]?.label} vs {AXIS[keys[1]]?.label}
                      </p>
                      <ScatterChart
                        data={data}
                        xKey={keys[0]} yKey={keys[1]}
                        onBrush={(f) => onBrush(f)}
                        width={540} height={220}
                      />
                    </div>
                  );
                }
                if (type === "histogram") {
                  return (
                    <div key={spec}>
                      <p className="text-xs tracking-widest uppercase opacity-35 mb-2">
                        {AXIS[keys[0]]?.label} distribution
                      </p>
                      <HistogramChart
                        data={data}
                        xKey={keys[0]}
                        onBrush={(f) => onBrush(f)}
                        width={540} height={165}
                      />
                    </div>
                  );
                }
                return null;
              })
            )}
            <p className="text-xs opacity-25 mt-auto">
              Brush scatter / click histogram bar to crossfilter across all nodes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── PipelineNode SVG component ────────────────────────────────────────────────
function PipelineNode({ node, active, onClick }) {
  const Sym = SYMBOLS[node.symbol];
  const c   = node.color;
  const glow = node.critical
    ? `drop-shadow(0 0 9px ${c}cc)`
    : active
    ? `drop-shadow(0 0 5px ${c}88)`
    : undefined;

  return (
    <g transform={`translate(${node.x},${node.y})`} onClick={onClick}
       style={{ cursor: "pointer" }}>
      <ellipse rx={42} ry={28} fill={active ? `${c}1a` : `${c}08`} />
      <rect x={-36} y={-22} width={72} height={44} rx={4}
            fill="rgba(4,8,10,0.97)" stroke={c}
            strokeWidth={node.critical ? 1.8 : 1}
            style={{ filter: glow }} />
      <g stroke={c} fill="none" transform={node.orient === "v" ? "rotate(90)" : ""}>
        <Sym c={c} />
      </g>
      <text y={32} textAnchor="middle" fontSize="7.5" fill={c} opacity="0.85"
            fontFamily="monospace" letterSpacing="0.06em">
        {node.label.toUpperCase()}
      </text>
      <text y={42} textAnchor="middle" fontSize="6" fill={c} opacity="0.38"
            fontFamily="monospace" letterSpacing="0.04em">
        {node.sub}
      </text>
    </g>
  );
}

// ── Flow arrow helper ─────────────────────────────────────────────────────────
function Arrow({ x, y, angle = 0 }) {
  return (
    <g transform={`translate(${x},${y}) rotate(${angle})`}>
      <polygon points="0,-4 8,0 0,4" fill="rgba(88,230,217,0.3)" />
    </g>
  );
}

// ── Circuit schematic ─────────────────────────────────────────────────────────
function CircuitSchematic({ selectedNode, onNodeClick, passCount, failCount, filteredLen }) {
  return (
    <svg viewBox="0 0 940 370" style={{ width: "100%", height: "100%" }}>
      {/* Outer wire rectangle */}
      <rect x={90} y={100} width={760} height={180}
            fill="none" stroke="rgba(88,230,217,0.22)" strokeWidth="1.5"
            strokeLinejoin="round" />

      {/* Flow direction arrows */}
      <Arrow x={185} y={100} angle={0}  />
      <Arrow x={385} y={100} angle={0}  />
      <Arrow x={580} y={100} angle={0}  />
      <Arrow x={787} y={100} angle={0}  />
      <Arrow x={850} y={165} angle={90} />
      <Arrow x={727} y={280} angle={180}/>
      <Arrow x={476} y={280} angle={180}/>
      <Arrow x={213} y={280} angle={180}/>
      <Arrow x={90}  y={213} angle={270}/>

      {/* Process flow label */}
      <text x={470} y={85} textAnchor="middle" fontSize="7.5"
            fill="rgba(88,230,217,0.28)" fontFamily="monospace" letterSpacing="0.12em">
        PROCESS FLOW
      </text>

      {/* Critical path label */}
      <text x={678} y={86} textAnchor="middle" fontSize="7"
            fill="rgba(245,158,11,0.55)" fontFamily="monospace" letterSpacing="0.08em">
        ★ CRITICAL
      </text>

      {/* Batch summary annotations */}
      <text x={580} y={360} textAnchor="middle" fontSize="8"
            fill="rgba(255,255,255,0.25)" fontFamily="monospace">
        {filteredLen} batches in filter · {passCount} passing (green) · {failCount} failing (red)
      </text>

      {/* Pipeline nodes */}
      {NODES.map(node => (
        <PipelineNode
          key={node.id}
          node={node}
          active={selectedNode?.id === node.id}
          onClick={() => onNodeClick(node)}
        />
      ))}
    </svg>
  );
}

// ── Filter status bar ─────────────────────────────────────────────────────────
function FilterBar({ filters, onClear }) {
  const active = Object.entries(filters).filter(([, r]) => r);
  if (!active.length) return null;
  return (
    <div
      className="flex items-center gap-4 px-8 py-2"
      style={{
        borderTop: "1px solid rgba(255,255,255,0.07)",
        background: "rgba(88,230,217,0.04)",
        fontSize: "0.62rem",
      }}
    >
      <span className="opacity-35 tracking-widest uppercase">Active filters:</span>
      {active.map(([k, r]) => (
        <span key={k} style={{ color: "#58E6D9" }}>
          {AXIS[k]?.label ?? k}: [{r[0].toFixed(1)}, {r[1].toFixed(1)}]
        </span>
      ))}
      <button onClick={onClear}
              className="ml-auto opacity-40 hover:opacity-80 transition-opacity tracking-widest uppercase">
        Clear ×
      </button>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function Ritonavir() {
  const [selectedNode, setSelectedNode] = useState(null);
  const [filters, setFilters]           = useState({});

  const filteredData = useMemo(() => applyFilters(BATCHES, filters), [filters]);
  const passCount    = filteredData.filter(d => d.passed).length;
  const failCount    = filteredData.length - passCount;

  const handleBrush = useCallback((f) => {
    setFilters(prev => f ? { ...prev, ...f } : {});
  }, []);

  const handleNodeClick = useCallback((node) => {
    setSelectedNode(prev => prev?.id === node.id ? null : node);
  }, []);

  return (
    <>
      <Head>
        <title>Ritonavir Synthesis — Categorical Spectrometry</title>
      </Head>

      <div className="w-full min-h-screen bg-dark text-light font-mont flex flex-col">
        {/* Header */}
        <header className="flex items-center justify-between px-8 py-5 border-b border-white/10">
          <Link href="/"
                className="text-xs tracking-widest uppercase opacity-40 hover:opacity-80 transition-opacity">
            ← Back
          </Link>
          <div className="text-center">
            <h1 className="text-sm tracking-widest uppercase font-medium opacity-80">
              Ritonavir Synthesis Pipeline
            </h1>
            <p className="text-xs mt-0.5 opacity-30 tracking-wider">
              Electric Circuit Schematic · Polymorph Control · Process Crossfiltering
            </p>
          </div>
          <div className="text-xs tracking-widest uppercase opacity-40 hover:opacity-80 transition-opacity">
            <Link href="/polymorphism">Spectral Hologram →</Link>
          </div>
        </header>

        {/* Instruction bar */}
        <div className="px-8 py-2.5"
             style={{ borderBottom: "1px solid rgba(255,255,255,0.05)", background: "rgba(255,255,255,0.01)" }}>
          <p className="text-xs opacity-30 tracking-wider">
            Click any node to inspect process parameters and crossfilter batch data.
            Brush scatter charts or click histogram bars to propagate filters across all nodes.
            {" "}{failCount > 0 && (
              <span style={{ color: "#f59e0b" }}>
                {failCount} failing batches — trace to Crystallization.
              </span>
            )}
          </p>
        </div>

        {/* Circuit schematic */}
        <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
          <CircuitSchematic
            selectedNode={selectedNode}
            onNodeClick={handleNodeClick}
            passCount={passCount}
            failCount={failCount}
            filteredLen={filteredData.length}
          />
        </div>

        {/* Filter bar */}
        <FilterBar filters={filters} onClear={() => setFilters({})} />
      </div>

      {/* Node modal */}
      {selectedNode && (
        <NodeModal
          node={selectedNode}
          data={filteredData}
          filters={filters}
          onBrush={handleBrush}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </>
  );
}
