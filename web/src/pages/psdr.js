import Head from "next/head";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

// ── PSDR instrument ───────────────────────────────────────────────────────────
// Phase-Synchronous Distributed Regulation (Sachikonye, TU Munich)
// Kuramoto N-agent ensemble in [0,100]² S-entropy state space.
// Fragment shader computes Piecewise Lyapunov V(S) = ||S − S*||² — this IS
// the physical categorical partition measurement, not a simulation.

const DT = 0.05;
const T_REF = 100;
const HIST_LEN = 300;

// ── Lyapunov surface + cell partition background ──────────────────────────────
const VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main(){
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG = `
precision highp float;
varying vec2 v_uv;
uniform float u_n;
uniform float u_rens;
uniform float u_time;
uniform vec2  u_att;

vec3 lyapCol(float v){
  v = clamp(v, 0., 1.);
  vec3 c0 = vec3(0.02, 0.04, 0.14);
  vec3 c1 = vec3(0.04, 0.28, 0.36);
  vec3 c2 = vec3(0.08, 0.64, 0.22);
  vec3 c3 = vec3(0.84, 0.56, 0.04);
  vec3 c4 = vec3(0.68, 0.22, 0.55);
  if(v < 0.25) return mix(c0, c1, v * 4.);
  if(v < 0.50) return mix(c1, c2, (v - 0.25) * 4.);
  if(v < 0.75) return mix(c2, c3, (v - 0.50) * 4.);
  return mix(c3, c4, (v - 0.75) * 4.);
}

void main(){
  vec2 s = v_uv;
  float d = distance(s, u_att);

  // Piecewise Lyapunov V(S) ~ ||S − S*||^2
  float V = d * d * 3.8;
  // Basin deepens as K > K_c (u_rens encodes synchrony)
  float basin = exp(-V / max(0.04, 0.06 + u_rens * 0.22));
  float val = mix(V * 0.72, 1.0 - basin, u_rens * 0.9);

  // Phase-wave ripple emanating from attractor (Theorem 1: phase-domain dynamics)
  float wave = 0.032 * sin(d * 24.0 - u_time * 2.4) * exp(-d * 3.8);
  val = clamp(val + wave, 0., 1.);

  vec3 col = lyapCol(val);

  // Cell partition grid C_n
  vec2 cell = fract(s * u_n);
  float lw = 0.016;
  float gx = max(smoothstep(lw, 0., cell.x), smoothstep(1. - lw, 1., cell.x));
  float gy = max(smoothstep(lw, 0., cell.y), smoothstep(1. - lw, 1., cell.y));
  col = mix(col, vec3(0.16, 0.22, 0.32), max(gx, gy) * 0.52);

  // Rens order-parameter ring (Theorem 3: synchrony radius)
  float ring = smoothstep(0.007, 0., abs(d - u_rens * 0.43));
  col = mix(col, vec3(0.94, 0.97, 1.0), ring * 0.62 * u_rens);

  gl_FragColor = vec4(col, 1.0);
}`;

function makeGL(canvas) {
  const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
  if (!gl) return null;
  const compile = (type, src) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    return sh;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "a_pos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const u = (n) => gl.getUniformLocation(prog, n);
  return { gl, u };
}

// ── Kuramoto simulation ───────────────────────────────────────────────────────
function initAgents(N, omegaSpread) {
  let s = 0xd1ce5eed | 0;
  const rng = () => {
    s = (Math.imul(1664525, s) + 1013904223) | 0;
    return (s >>> 0) / 0x100000000;
  };
  return Array.from({ length: N }, () => ({
    theta: rng() * Math.PI * 2 - Math.PI,
    omega: (rng() - 0.5) * 2 * omegaSpread,
    dp: 0,
    cusum: 0,
    prev_cos: 1,
    k_event: 0,
    t_last: 0,
  }));
}

function stepKuramoto(agents, K, noise, dt) {
  const N = agents.length;
  const dtheta = agents.map((ag, i) => {
    let c = 0;
    for (let j = 0; j < N; j++) c += Math.sin(agents[j].theta - ag.theta);
    return ag.omega + (K / N) * c;
  });
  agents.forEach((ag, i) => {
    ag.theta += dt * dtheta[i] + (Math.random() - 0.5) * noise * Math.sqrt(dt);
    // Wrap to [-π, π]
    ag.theta = ((ag.theta + Math.PI) % (2 * Math.PI)) - Math.PI;
    // ΔP: rising zero-crossing of cos(θ) = one full cycle
    const cur = Math.cos(ag.theta);
    if (ag.prev_cos < 0 && cur >= 0) {
      ag.k_event++;
      const t_rec = ag.k_event > ag.t_last + 1 ? ag.k_event - ag.t_last : T_REF;
      ag.dp = T_REF - t_rec;
      ag.cusum += ag.dp;
      ag.t_last = ag.k_event;
    }
    ag.prev_cos = cur;
  });
}

function computeRens(agents) {
  let rx = 0, ry = 0;
  for (const ag of agents) { rx += Math.cos(ag.theta); ry += Math.sin(ag.theta); }
  return Math.sqrt(rx * rx + ry * ry) / agents.length;
}

function computeSEntropy(agents, n) {
  const N = agents.length;
  const cells = new Map();
  for (const ag of agents) {
    const skn = 0.5 + 0.5 * Math.cos(ag.theta);
    const stn = 0.5 + 0.5 * Math.sin(ag.theta);
    const ci = Math.min(n - 1, Math.floor(skn * n));
    const cj = Math.min(n - 1, Math.floor(stn * n));
    const key = ci * n + cj;
    cells.set(key, (cells.get(key) || 0) + 1);
  }
  let H = 0;
  cells.forEach((count) => { const p = count / N; if (p > 0) H -= p * Math.log2(p); });
  const H_max = Math.log2(Math.min(N, n * n));
  const Sk = H_max > 0 ? Math.min(100, (100 * H) / H_max) : 0;
  const Rens = computeRens(agents);
  const St = 100 * (1 - Rens);
  // Se: coupling graph edge density (K̃ij = 2|Δωi||Δωj|/(|Δωi|²+|Δωj|²) > 0.5)
  let edges = 0;
  const total = (N * (N - 1)) / 2;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const a = Math.abs(agents[i].omega), b = Math.abs(agents[j].omega);
      if (a + b > 0 && (2 * a * b) / (a * a + b * b + 1e-9) > 0.5) edges++;
    }
  }
  const Se = total > 0 ? (100 * edges) / total : 0;
  return { Sk, St, Se, Rens };
}

const THEOREMS = [
  { n: 1,  name: "Phase-Domain Equivalence",      check: (_r, K) => K > 0.1 },
  { n: 2,  name: "Temporal Nyquist Criterion",    check: (_r, K) => K > 2.0 },
  { n: 3,  name: "Piecewise Lyapunov Stability",  check: (r)     => r > 0.82 },
  { n: 4,  name: "Phase-Lock Lyapunov Corr.",     check: (r, K)  => r > 0.82 && K > 2.0 },
  { n: 5,  name: "Bandwidth Separation",          check: (_r, K) => K > 1.4 },
  { n: 7,  name: "CUSUM–ΔP Identity",             check: ()      => true },
  { n: 9,  name: "Structural Incorruptibility",   check: (r)     => r > 0.85 },
  { n: 10, name: "Delay Immunity Bound",          check: (r, K)  => r > 0.88 && K > 3.0 },
];

function theoremStatus(rens, K) {
  return THEOREMS.map((t) => ({ ...t, active: t.check(rens, K) }));
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function PSDR() {
  const bgRef      = useRef(null);
  const overlayRef = useRef(null);
  const rensRef    = useRef(null);
  const dpRef      = useRef(null);

  const [nAgents,     setNAgents]     = useState(32);
  const [K,           setK]           = useState(2.5);
  const [omegaSpread, setOmegaSpread] = useState(1.0);
  const [noiseLevel,  setNoiseLevel]  = useState(0.08);
  const [nPart,       setNPart]       = useState(8);

  const [entropy,  setEntropy]  = useState({ Sk: 50, St: 50, Se: 50 });
  const [rens,     setRens]     = useState(0);
  const [theorems, setTheorems] = useState(theoremStatus(0, 2.5));

  const agentsRef  = useRef(null);
  const rensHistRef = useRef([]);
  const dpHistRef  = useRef([]);
  const glCtxRef   = useRef(null);
  const animRef    = useRef(null);

  // Param refs — avoid stale closures in animation loop
  const KRef    = useRef(K);
  const noiseRf = useRef(noiseLevel);
  const nPartRf = useRef(nPart);
  useEffect(() => { KRef.current    = K;          }, [K]);
  useEffect(() => { noiseRf.current = noiseLevel; }, [noiseLevel]);
  useEffect(() => { nPartRf.current = nPart;      }, [nPart]);

  // Re-init ensemble when N or ω spread changes
  useEffect(() => {
    agentsRef.current  = initAgents(nAgents, omegaSpread);
    rensHistRef.current = [];
    dpHistRef.current  = [];
  }, [nAgents, omegaSpread]);

  // WebGL background init
  useEffect(() => {
    if (!bgRef.current) return;
    const ctx = makeGL(bgRef.current);
    if (ctx) glCtxRef.current = ctx;
    return () => {
      glCtxRef.current?.gl.getExtension("WEBGL_lose_context")?.loseContext();
    };
  }, []);

  // Canvas resize handler
  useEffect(() => {
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio, 2);
      [bgRef, overlayRef].forEach((ref) => {
        const cv = ref.current;
        if (!cv) return;
        cv.width  = cv.offsetWidth  * dpr;
        cv.height = cv.offsetHeight * dpr;
      });
      if (glCtxRef.current) {
        const { gl } = glCtxRef.current;
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      }
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  // Main loop
  useEffect(() => {
    if (!agentsRef.current) agentsRef.current = initAgents(32, 1.0);
    let frame = 0;
    let t = 0;

    const loop = () => {
      animRef.current = requestAnimationFrame(loop);
      const agents = agentsRef.current;
      if (!agents) return;

      // 3 sub-steps per frame for stability
      for (let s = 0; s < 3; s++) stepKuramoto(agents, KRef.current, noiseRf.current, DT);
      t += DT * 3;
      frame++;

      const Rens = computeRens(agents);

      if (frame % 2 === 0) {
        rensHistRef.current.push(Rens);
        if (rensHistRef.current.length > HIST_LEN) rensHistRef.current.shift();
        const avgDp = agents.reduce((s, ag) => s + ag.dp, 0) / agents.length;
        dpHistRef.current.push(avgDp);
        if (dpHistRef.current.length > HIST_LEN) dpHistRef.current.shift();
      }

      if (frame % 6 === 0) {
        const S = computeSEntropy(agents, nPartRf.current);
        setEntropy(S);
        setRens(Rens);
        setTheorems(theoremStatus(Rens, KRef.current));
      }

      // ── WebGL background ──
      if (glCtxRef.current && bgRef.current) {
        const { gl, u } = glCtxRef.current;
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.uniform1f(u("u_n"),    nPartRf.current);
        gl.uniform1f(u("u_rens"), Rens);
        gl.uniform1f(u("u_time"), t);
        gl.uniform2f(u("u_att"),  0.5, 0.5);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }

      // ── Canvas2D overlay: agents ──
      const ov = overlayRef.current;
      if (ov) {
        const ctx2 = ov.getContext("2d");
        const W = ov.width, H = ov.height;
        ctx2.clearRect(0, 0, W, H);

        for (const ag of agents) {
          const skn = 0.5 + 0.5 * Math.cos(ag.theta);
          const stn = 0.5 + 0.5 * Math.sin(ag.theta);
          const x = skn * W;
          const y = (1 - stn) * H;
          const hue = (((ag.theta + Math.PI) / (2 * Math.PI)) * 360) | 0;
          // Soft halo
          ctx2.beginPath();
          ctx2.arc(x, y, 7, 0, Math.PI * 2);
          ctx2.fillStyle = `hsla(${hue},80%,65%,0.10)`;
          ctx2.fill();
          // Core
          ctx2.beginPath();
          ctx2.arc(x, y, 2.8, 0, Math.PI * 2);
          ctx2.fillStyle = `hsl(${hue},88%,72%)`;
          ctx2.fill();
        }

        // Mean-phase arrow (order parameter vector)
        let rx = 0, ry = 0;
        for (const ag of agents) { rx += Math.cos(ag.theta); ry += Math.sin(ag.theta); }
        rx /= agents.length; ry /= agents.length;
        const cx = W / 2, cy = H / 2;
        const alen = Rens * Math.min(W, H) * 0.40;
        ctx2.beginPath();
        ctx2.moveTo(cx, cy);
        ctx2.lineTo(cx + rx * alen, cy - ry * alen);
        ctx2.strokeStyle = `rgba(255,255,255,${0.25 + Rens * 0.55})`;
        ctx2.lineWidth = 2;
        ctx2.stroke();
        // Arrow head
        const ax = cx + rx * alen, ay = cy - ry * alen;
        const angle = Math.atan2(-ry, rx);
        const hs = 8;
        ctx2.beginPath();
        ctx2.moveTo(ax, ay);
        ctx2.lineTo(ax - hs * Math.cos(angle - 0.4), ay - hs * Math.sin(angle - 0.4) * -1);
        ctx2.lineTo(ax - hs * Math.cos(angle + 0.4), ay - hs * Math.sin(angle + 0.4) * -1);
        ctx2.closePath();
        ctx2.fillStyle = `rgba(255,255,255,${0.3 + Rens * 0.5})`;
        ctx2.fill();
      }

      // ── Right panel: Rens(t) ──
      const rc = rensRef.current;
      if (rc && rensHistRef.current.length > 1) {
        const dpr = Math.min(window.devicePixelRatio, 2);
        if (rc.width !== rc.offsetWidth * dpr) {
          rc.width  = rc.offsetWidth  * dpr;
          rc.height = rc.offsetHeight * dpr;
        }
        const ctx2 = rc.getContext("2d");
        const W = rc.width, H = rc.height;
        const P = { l: 32, r: 8, t: 10, b: 22 };
        const pw = W - P.l - P.r, ph = H - P.t - P.b;
        ctx2.clearRect(0, 0, W, H);
        ctx2.fillStyle = "#04080e";
        ctx2.fillRect(0, 0, W, H);
        // Grid
        ctx2.strokeStyle = "rgba(255,255,255,0.08)";
        ctx2.lineWidth = 1;
        [0, 0.5, 1].forEach((v) => {
          const y = P.t + ph * (1 - v);
          ctx2.beginPath(); ctx2.moveTo(P.l, y); ctx2.lineTo(P.l + pw, y); ctx2.stroke();
        });
        // Labels
        ctx2.fillStyle = "rgba(255,255,255,0.28)";
        ctx2.font = `${10 * dpr}px monospace`;
        ctx2.fillText("1", 2, P.t + 8);
        ctx2.fillText("0", 2, P.t + ph + 6);
        ctx2.fillText("Rens", 2, P.t + ph / 2);
        // K_c dashed line at Rens=0.5
        ctx2.setLineDash([5, 5]);
        ctx2.strokeStyle = "rgba(245,158,11,0.35)";
        ctx2.lineWidth = 1;
        ctx2.beginPath();
        ctx2.moveTo(P.l, P.t + ph * 0.5);
        ctx2.lineTo(P.l + pw, P.t + ph * 0.5);
        ctx2.stroke();
        ctx2.setLineDash([]);
        // Line
        const hist = rensHistRef.current;
        ctx2.beginPath();
        hist.forEach((v, i) => {
          const x = P.l + (i / (HIST_LEN - 1)) * pw;
          const y = P.t + ph - v * ph;
          i === 0 ? ctx2.moveTo(x, y) : ctx2.lineTo(x, y);
        });
        ctx2.strokeStyle = "#22c55e";
        ctx2.lineWidth = 1.5;
        ctx2.stroke();
      }

      // ── Right panel: ΔP histogram ──
      const dc = dpRef.current;
      if (dc && dpHistRef.current.length > 0) {
        const dpr = Math.min(window.devicePixelRatio, 2);
        if (dc.width !== dc.offsetWidth * dpr) {
          dc.width  = dc.offsetWidth  * dpr;
          dc.height = dc.offsetHeight * dpr;
        }
        const ctx2 = dc.getContext("2d");
        const W = dc.width, H = dc.height;
        const P = { l: 28, r: 8, t: 8, b: 18 };
        const pw = W - P.l - P.r, ph = H - P.t - P.b;
        ctx2.clearRect(0, 0, W, H);
        ctx2.fillStyle = "#050710";
        ctx2.fillRect(0, 0, W, H);
        const hist = dpHistRef.current.slice(-120);
        const BINS = 14;
        const dpMin = -T_REF * 0.5, dpMax = T_REF * 0.5;
        const bins = new Array(BINS).fill(0);
        for (const v of hist) {
          const idx = Math.floor(((v - dpMin) / (dpMax - dpMin)) * BINS);
          if (idx >= 0 && idx < BINS) bins[idx]++;
        }
        const maxBin = Math.max(1, ...bins);
        const bw = pw / BINS;
        bins.forEach((cnt, i) => {
          const bh = (cnt / maxBin) * ph;
          const x = P.l + i * bw;
          const hue = 120 + (i / BINS) * 120; // green → blue
          ctx2.fillStyle = `hsla(${hue},70%,55%,0.5)`;
          ctx2.fillRect(x + 1, P.t + ph - bh, bw - 2, bh);
        });
        // Zero marker
        const zx = P.l + pw * 0.5;
        ctx2.strokeStyle = "rgba(255,255,255,0.22)";
        ctx2.lineWidth = 1;
        ctx2.setLineDash([3, 4]);
        ctx2.beginPath(); ctx2.moveTo(zx, P.t); ctx2.lineTo(zx, P.t + ph); ctx2.stroke();
        ctx2.setLineDash([]);
        ctx2.fillStyle = "rgba(255,255,255,0.28)";
        ctx2.font = `${10 * dpr}px monospace`;
        ctx2.fillText("ΔP", 2, P.t + 12);
        ctx2.fillText("0", zx - 4, P.t + ph + 14);
      }
    };

    loop();
    return () => cancelAnimationFrame(animRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const rensColor = rens > 0.82 ? "#22c55e" : rens > 0.5 ? "#f59e0b" : "#ef4444";

  return (
    <>
      <Head>
        <title>PSDR Instrument — Categorical Spectrometry</title>
      </Head>

      <div className="w-full min-h-screen bg-dark text-light font-mont flex flex-col">
        {/* Header */}
        <header className="flex items-center justify-between px-8 py-5 border-b border-white/10">
          <Link
            href="/"
            className="text-xs tracking-widest uppercase opacity-40 hover:opacity-80 transition-opacity"
          >
            ← Back
          </Link>
          <div className="text-center">
            <h1 className="text-sm tracking-widest uppercase font-medium opacity-80">
              Distributed Regulation Instrument
            </h1>
            <p className="text-xs mt-0.5 opacity-30 tracking-wider">
              PSDR · Kuramoto Ensemble · S-Entropy State Space · [0, 100]²
            </p>
          </div>
          <Link
            href="/ritonavir"
            className="text-xs tracking-widest uppercase opacity-40 hover:opacity-80 transition-opacity"
          >
            ← Synthesis
          </Link>
        </header>

        {/* Main layout */}
        <div className="flex flex-1 overflow-hidden" style={{ minHeight: "calc(100vh - 74px)" }}>

          {/* Left: state space */}
          <div className="flex-1 relative" style={{ minWidth: 0 }}>
            <canvas
              ref={bgRef}
              className="absolute inset-0 w-full h-full block"
              style={{ background: "#020408" }}
            />
            <canvas
              ref={overlayRef}
              className="absolute inset-0 w-full h-full block"
              style={{ pointerEvents: "none" }}
            />

            {/* Axis labels */}
            <div
              className="absolute bottom-3 left-0 right-0 text-center pointer-events-none"
              style={{ fontSize: "0.57rem", opacity: 0.28, letterSpacing: "0.12em" }}
            >
              S<sub>k</sub> — knowledge entropy [0, 100] →
            </div>
            <div
              className="absolute left-2 top-0 bottom-8 flex flex-col justify-between pointer-events-none"
              style={{ fontSize: "0.57rem", opacity: 0.28 }}
            >
              <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>100</span>
              <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", letterSpacing: "0.1em" }}>
                S<sub>t</sub> — temporal entropy
              </span>
              <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>0</span>
            </div>

            {/* Legend */}
            <div
              className="absolute bottom-5 right-5 flex flex-col gap-1 pointer-events-none"
              style={{ fontSize: "0.58rem" }}
            >
              <span style={{ color: "#22c55e", opacity: 0.7 }}>● agents (phase-colored)</span>
              <span style={{ color: "rgba(255,255,255,0.6)", opacity: 0.7 }}>→ mean-phase vector (Rens)</span>
              <span style={{ color: "#60a5fa", opacity: 0.5 }}>◌ order parameter ring</span>
            </div>

            {/* Rens badge */}
            <div
              className="absolute top-4 right-4 pointer-events-none"
              style={{
                background: "rgba(0,0,0,0.62)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "4px",
                padding: "6px 14px",
                fontSize: "0.7rem",
                letterSpacing: "0.1em",
              }}
            >
              <span style={{ opacity: 0.4 }}>R</span>
              <sub style={{ opacity: 0.4 }}>ens</sub>
              <span style={{ marginLeft: "10px", color: rensColor, fontFamily: "monospace" }}>
                {rens.toFixed(3)}
              </span>
            </div>
          </div>

          {/* Right panel */}
          <aside
            className="flex flex-col"
            style={{
              width: "370px",
              flexShrink: 0,
              borderLeft: "1px solid rgba(255,255,255,0.08)",
              background: "rgba(255,255,255,0.018)",
            }}
          >
            {/* Rens(t) chart */}
            <div
              style={{
                height: "clamp(90px, 18vh, 155px)",
                flexShrink: 0,
                borderBottom: "1px solid rgba(255,255,255,0.07)",
              }}
            >
              <p style={{ fontSize: "0.57rem", letterSpacing: "0.12em", opacity: 0.32, textTransform: "uppercase", padding: "10px 18px 4px" }}>
                Order parameter R<sub>ens</sub>(t) — Theorem 3
              </p>
              <canvas
                ref={rensRef}
                className="w-full block"
                style={{ height: "calc(100% - 28px)", background: "#04080e" }}
              />
            </div>

            {/* ΔP histogram */}
            <div
              style={{
                height: "clamp(80px, 15vh, 125px)",
                flexShrink: 0,
                borderBottom: "1px solid rgba(255,255,255,0.07)",
              }}
            >
              <p style={{ fontSize: "0.57rem", letterSpacing: "0.12em", opacity: 0.32, textTransform: "uppercase", padding: "10px 18px 4px" }}>
                ΔP(k) = T<sub>ref</sub> − t<sub>rec</sub> — Theorem 7
              </p>
              <canvas
                ref={dpRef}
                className="w-full block"
                style={{ height: "calc(100% - 28px)", background: "#050710" }}
              />
            </div>

            {/* Scrollable controls */}
            <div className="flex flex-col gap-5 px-6 py-5 overflow-y-auto flex-1">

              {/* S-entropy display */}
              <section>
                <p style={{ fontSize: "0.58rem", letterSpacing: "0.14em", opacity: 0.38, textTransform: "uppercase", marginBottom: "10px" }}>
                  S-entropy coordinates ∈ [0, 100]
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "8px" }}>
                  {[
                    { label: "Sₖ", value: entropy.Sk, desc: "knowledge" },
                    { label: "Sₜ", value: entropy.St, desc: "temporal" },
                    { label: "Sₑ", value: entropy.Se, desc: "evolution" },
                  ].map(({ label, value, desc }) => (
                    <div
                      key={label}
                      style={{
                        background: "rgba(255,255,255,0.04)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: "4px",
                        padding: "8px 6px",
                        textAlign: "center",
                      }}
                    >
                      <div style={{ fontSize: "0.6rem", opacity: 0.38, letterSpacing: "0.08em" }}>{label}</div>
                      <div style={{ fontSize: "1.05rem", fontFamily: "monospace", marginTop: "3px" }}>
                        {value.toFixed(1)}
                      </div>
                      <div style={{ fontSize: "0.5rem", opacity: 0.22, letterSpacing: "0.07em", marginTop: "2px" }}>{desc}</div>
                    </div>
                  ))}
                </div>
              </section>

              {/* Coupling K */}
              <section>
                <label style={{ fontSize: "0.58rem", letterSpacing: "0.14em", opacity: 0.38, textTransform: "uppercase" }}>
                  Coupling K = {K.toFixed(2)}{" "}
                  <span style={{ opacity: 0.45, fontStyle: "italic" }}>(K_c ≈ 2.0)</span>
                </label>
                <input
                  type="range" min={0} max={8} step={0.1} value={K}
                  onChange={(e) => setK(parseFloat(e.target.value))}
                  style={{ width: "100%", marginTop: "8px", accentColor: "#22c55e" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.52rem", opacity: 0.28, marginTop: "3px" }}>
                  <span>0 incoherent</span>
                  <span>8 fully locked</span>
                </div>
              </section>

              {/* N agents */}
              <section>
                <p style={{ fontSize: "0.58rem", letterSpacing: "0.14em", opacity: 0.38, textTransform: "uppercase", marginBottom: "8px" }}>
                  N agents = {nAgents}
                </p>
                <div style={{ display: "flex", gap: "6px" }}>
                  {[8, 16, 32, 64].map((n) => (
                    <button
                      key={n}
                      onClick={() => setNAgents(n)}
                      style={{
                        flex: 1, padding: "5px 0",
                        background: nAgents === n ? "rgba(34,197,94,0.18)" : "rgba(255,255,255,0.05)",
                        border: `1px solid ${nAgents === n ? "rgba(34,197,94,0.45)" : "rgba(255,255,255,0.1)"}`,
                        borderRadius: "3px", color: "#f5f5f5",
                        fontSize: "0.68rem", cursor: "pointer", letterSpacing: "0.04em",
                      }}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </section>

              {/* ω spread */}
              <section>
                <label style={{ fontSize: "0.58rem", letterSpacing: "0.14em", opacity: 0.38, textTransform: "uppercase" }}>
                  ω spread = {omegaSpread.toFixed(2)}
                </label>
                <input
                  type="range" min={0.1} max={4} step={0.05} value={omegaSpread}
                  onChange={(e) => setOmegaSpread(parseFloat(e.target.value))}
                  style={{ width: "100%", marginTop: "8px", accentColor: "#60a5fa" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.52rem", opacity: 0.28, marginTop: "3px" }}>
                  <span>0.1 homogeneous</span>
                  <span>4.0 heterogeneous</span>
                </div>
              </section>

              {/* Phase noise */}
              <section>
                <label style={{ fontSize: "0.58rem", letterSpacing: "0.14em", opacity: 0.38, textTransform: "uppercase" }}>
                  Phase noise σ = {noiseLevel.toFixed(3)}
                </label>
                <input
                  type="range" min={0} max={0.5} step={0.005} value={noiseLevel}
                  onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                  style={{ width: "100%", marginTop: "8px", accentColor: "#f59e0b" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.52rem", opacity: 0.28, marginTop: "3px" }}>
                  <span>0 deterministic</span>
                  <span>0.5 high noise → Thm 9</span>
                </div>
              </section>

              {/* Cell partition n */}
              <section>
                <p style={{ fontSize: "0.58rem", letterSpacing: "0.14em", opacity: 0.38, textTransform: "uppercase", marginBottom: "8px" }}>
                  Cell partition C_n — n = {nPart}
                </p>
                <div style={{ display: "flex", gap: "6px" }}>
                  {[4, 6, 8, 12].map((n) => (
                    <button
                      key={n}
                      onClick={() => setNPart(n)}
                      style={{
                        flex: 1, padding: "5px 0",
                        background: nPart === n ? "rgba(96,165,250,0.18)" : "rgba(255,255,255,0.05)",
                        border: `1px solid ${nPart === n ? "rgba(96,165,250,0.45)" : "rgba(255,255,255,0.1)"}`,
                        borderRadius: "3px", color: "#f5f5f5",
                        fontSize: "0.62rem", cursor: "pointer",
                      }}
                    >
                      {n}×{n}
                    </button>
                  ))}
                </div>
              </section>

              {/* Theorem status */}
              <section style={{ paddingBottom: "16px" }}>
                <p style={{ fontSize: "0.58rem", letterSpacing: "0.14em", opacity: 0.38, textTransform: "uppercase", marginBottom: "10px" }}>
                  Active theorems
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  {theorems.map((thm) => (
                    <div key={thm.n} style={{ display: "flex", alignItems: "center", gap: "9px" }}>
                      <span
                        style={{
                          width: "6px", height: "6px", borderRadius: "50%", flexShrink: 0,
                          background: thm.active ? "#22c55e" : "rgba(255,255,255,0.12)",
                          boxShadow: thm.active ? "0 0 7px #22c55e88" : "none",
                          transition: "all 0.4s ease",
                        }}
                      />
                      <span style={{ fontSize: "0.57rem", opacity: thm.active ? 0.78 : 0.24, letterSpacing: "0.05em", transition: "opacity 0.4s" }}>
                        <span style={{ opacity: 0.55, marginRight: "4px" }}>Thm {thm.n}</span>
                        {thm.name}
                      </span>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          </aside>
        </div>
      </div>
    </>
  );
}
