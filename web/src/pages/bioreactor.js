import Head from "next/head";
import Link from "next/link";
import { useEffect, useRef, useState, useCallback } from "react";

// ─── Bioreactor metabolic hologram ───────────────────────────────────────────
// Computes H(ω, φ_ATP) = Σᵢ Sᵢ(ω)·cos(φ_ATP + δᵢ) for ATP-constrained dynamics.
// This IS the GPU-Interference Isomorphism: the fragment shader computing
// I_int = (I_A + I_B + 2√(I_A·I_B)·cos(Δφ))/2 for metabolic flux interference
// is physically identical to the measurement.  ATP drives δᵢ.
// Evolution as dx/d[ATP] — ATP is the time coordinate (mogadishu framework).

const VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main(){
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// Main canvas: metabolic phase-frequency hologram
// X = metabolite oscillation frequency (5 bands: Glu, Lac, Gln, O2, ATP/ADP)
// Y = ATP synthesis phase angle [0, 2π]
const FRAG_HOLO = `
precision highp float;
varying vec2 v_uv;
uniform float u_time;
uniform float u_atp;      // 0.0–1.0 normalised (maps 0.5–15 mM)
uniform float u_phase;    // 0=exponential  1=decline
uniform float u_w_gln;
uniform float u_w_O2;

float mm (float a, float km){ return a / (a + km); }
float mmI(float a, float km){ return km / (a + km); }
float gpk(float x, float mu, float sg){
  float d = (x - mu) / sg;
  return exp(-0.5 * d * d);
}

// Michaelis-Menten metabolic amplitudes
float aGlu(float a, float ph){ return (0.88 * mmI(a, 0.30) + 0.12) * (1. - 0.65 * ph); }
float aLac(float a, float ph){ return  0.72 * mmI(a, 0.48) * (1. + ph * 1.15); }
float aGln(float a, float ph){ return  0.52 * mm(a, 0.40) * u_w_gln * (1. - 0.25 * ph); }
float aO2 (float a, float ph){ return  0.84 * mm(a, 0.58) * u_w_O2  * (1. - 0.48 * ph); }
float aATP(float a)           { return  a * 0.55 + 0.12; }

// Bio false-color: dark → forest-green → lime → amber → magenta
vec3 bioCol(float t){
  vec3 c0 = vec3(0.03, 0.06, 0.04);
  vec3 c1 = vec3(0.05, 0.22, 0.10);
  vec3 c2 = vec3(0.13, 0.78, 0.26);
  vec3 c3 = vec3(0.88, 0.60, 0.05);
  vec3 c4 = vec3(0.71, 0.24, 0.59);
  t = clamp(t, 0., 1.);
  if(t < 0.25) return mix(c0, c1, t * 4.);
  if(t < 0.50) return mix(c1, c2, (t - 0.25) * 4.);
  if(t < 0.75) return mix(c2, c3, (t - 0.50) * 4.);
  return mix(c3, c4, (t - 0.75) * 4.);
}

void main(){
  float f   = v_uv.x;
  float phi = v_uv.y * 6.28318;
  float a   = u_atp;
  float ph  = u_phase;

  // Metabolic "spectrum" — Gaussian peaks in frequency space
  float Sg  = aGlu(a, ph) * gpk(f, 0.13, 0.054);
  float Sl  = aLac(a, ph) * gpk(f, 0.30, 0.065);
  float Sgn = aGln(a, ph) * gpk(f, 0.52, 0.068);
  float So  = aO2 (a, ph) * gpk(f, 0.73, 0.054);
  float Sa  = aATP(a)     * gpk(f, 0.88, 0.044);

  // Phase offsets encode ATP-coupling (electrochemical gradient driven)
  float dLac = 0.46 * 3.14159 * (1. - a);   // lactate lags glucose at low ATP
  float dGln = 0.80 * 3.14159;               // glutamine nearly anti-phase (anaplerotic)
  float dO2  = 1.18 * 3.14159 * a;           // O2 phase follows ATP availability
  float dATP = 0.24 * 3.14159;               // ATP/ADP slight lead

  float te = u_time * mix(0.68, 0.22, ph);  // slow in decline

  // Metabolic hologram: ATP-phase-resolved interference
  float Hr = Sg  * cos(phi + te)
           + Sl  * cos(phi + dLac + te * 1.33)
           + Sgn * cos(phi + dGln + te * 0.64)
           + So  * cos(phi + dO2  + te * 1.68)
           + Sa  * cos(phi + dATP + te * 2.02);
  float Hi = Sg  * sin(phi + te)
           + Sl  * sin(phi + dLac + te * 1.33)
           + Sgn * sin(phi + dGln + te * 0.64)
           + So  * sin(phi + dO2  + te * 1.68)
           + Sa  * sin(phi + dATP + te * 2.02);

  float intensity = sqrt(Hr * Hr + Hi * Hi);

  // Spectrum band overlay at top (last 12%)
  float bs = 0.88;
  float inB = smoothstep(bs, bs + 0.04, v_uv.y);
  float bp  = (v_uv.y - bs) / (1. - bs);

  vec3 col = bioCol(intensity * 0.74);

  if(inB > 0.) {
    float bG  = Sg  * 3. * smoothstep(0.,    0.28, bp) * (1. - smoothstep(0.28, 0.40, bp));
    float bL  = Sl  * 3. * smoothstep(0.28,  0.44, bp) * (1. - smoothstep(0.44, 0.56, bp));
    float bGn = Sgn * 3. * smoothstep(0.56,  0.70, bp) * (1. - smoothstep(0.70, 0.82, bp));
    float bO  = So  * 3. * smoothstep(0.82,  0.95, bp);
    col = mix(col,
      vec3(0.15,0.90,0.28)*bG + vec3(0.90,0.25,0.15)*bL +
      vec3(0.25,0.60,1.00)*bGn + vec3(0.85,0.85,1.00)*bO + col*0.08,
      inB);
  }

  gl_FragColor = vec4(col, 1.);
}`;

// ATP flux canvas: d[ATP]/dt vs metabolite frequency
// Projection of the hologram onto the frequency axis — same X as main canvas
const FRAG_FLUX = `
precision highp float;
varying vec2 v_uv;
uniform float u_atp;
uniform float u_phase;
uniform float u_w_gln;
uniform float u_w_O2;

float mm (float a, float km){ return a / (a + km); }
float mmI(float a, float km){ return km / (a + km); }
float gpk(float x, float mu, float sg){ float d=(x-mu)/sg; return exp(-0.5*d*d); }

void main(){
  float f  = v_uv.x;
  float a  = u_atp;
  float ph = u_phase;

  float gly  = 2.0  * (mmI(a,0.30)+0.12) * (1.-ph*0.65)  * gpk(f,0.13,0.054);
  float lac  =-0.4  *  mmI(a,0.48)        * (1.+ph*1.15)  * gpk(f,0.30,0.065);
  float gln  = 4.5  *  mm(a,0.40) * u_w_gln*(1.-ph*0.25)  * gpk(f,0.52,0.068);
  float oxph = 29.0 *  mm(a,0.58) * u_w_O2 *(1.-ph*0.48)  * gpk(f,0.73,0.054);
  float atpR = 1.0  * (a*0.55+0.12)                        * gpk(f,0.88,0.044);

  float R = clamp((gly + lac + gln + oxph + atpR) / 31.0, -0.05, 1.0);
  float Rn = R * 0.86 + 0.07;

  vec3 bg = vec3(0.04, 0.07, 0.05) + 0.012 * step(0.97, mod(v_uv.y * 4., 1.));

  vec3 fCol;
  if(f < 0.22)      fCol = vec3(0.15, 0.90, 0.28);
  else if(f < 0.42) fCol = vec3(0.90, 0.25, 0.15);
  else if(f < 0.63) fCol = vec3(0.25, 0.60, 1.00);
  else if(f < 0.82) fCol = vec3(0.85, 0.85, 1.00);
  else              fCol = vec3(0.88, 0.60, 0.05);

  float onLine = smoothstep(0.011, 0., abs(v_uv.y - Rn));
  float fill   = step(v_uv.y, Rn) * 0.3;
  float axisH  = smoothstep(0.006, 0., abs(v_uv.y - 0.07));

  vec3 col = bg + fCol * (onLine + fill) + vec3(0.15, 0.25, 0.15) * axisH;
  gl_FragColor = vec4(clamp(col, 0., 1.), 1.);
}`;

// ─── WebGL helpers ────────────────────────────────────────────────────────────

function makeGL(canvas, fragSrc) {
  const gl = canvas.getContext("webgl");
  if (!gl) return null;
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fragSrc));
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

// ─── Metabolic analytics ──────────────────────────────────────────────────────

const METABOLITES = ["Glucose", "Lactate", "Glutamine", "O₂", "ATP/ADP"];
const MET_SHORT   = ["Glu", "Lac", "Gln", "O₂", "ATP"];

function computeFluxes(atp, phase, wGln, wO2) {
  const mm  = (a, km) => a / (a + km);
  const mmI = (a, km) => km / (a + km);
  return [
    (0.88 * mmI(atp, 0.30) + 0.12) * (1 - 0.65 * phase),
     0.72 * mmI(atp, 0.48) * (1 + phase * 1.15),
     0.52 *  mm(atp, 0.40) * wGln * (1 - 0.25 * phase),
     0.84 *  mm(atp, 0.58) * wO2  * (1 - 0.48 * phase),
     atp * 0.55 + 0.12,
  ];
}

function computeSEntropy(atp, phase, wGln, wO2) {
  const fluxes = computeFluxes(atp, phase, wGln, wO2);
  const total  = fluxes.reduce((a, b) => a + Math.abs(b), 0) || 1;
  const pn     = fluxes.map((v) => Math.abs(v) / total);
  const H      = -pn.reduce((s, p) => s + (p > 1e-10 ? p * Math.log(p) : 0), 0);
  const Sk     = Math.min(1, H / Math.log(5));

  // St: log(τ_gen / τ_ATP) / log(τ_gen / τ_P)
  const tau_ATP = 0.01 / (atp + 0.1);             // faster cycling at higher [ATP]
  const tau_gen = 72000 * (1 + phase * 1.8);       // generation time lengthens in decline
  const tau_P   = 5.39e-44;
  const St = Math.log(tau_gen / tau_ATP) / Math.log(tau_gen / tau_P);

  // Se: coupling K̃ᵢⱼ edge density (edges where K > 0.6)
  let edges = 0;
  const M = fluxes.length;
  for (let i = 0; i < M; i++)
    for (let j = i + 1; j < M; j++) {
      const a = Math.abs(fluxes[i]), b = Math.abs(fluxes[j]);
      if (a < 1e-4 || b < 1e-4) continue;
      if ((2 * a * b) / (a * a + b * b) > 0.6) edges++;
    }
  const Se = (2 * edges) / (M * (M - 1));

  return {
    Sk: Math.min(1, Math.max(0, Sk)),
    St: Math.min(1, Math.max(0, St)),
    Se: Math.min(1, Math.max(0, Se)),
  };
}

function computeCouplingMatrix(atp, phase, wGln, wO2) {
  const fluxes = computeFluxes(atp, phase, wGln, wO2);
  return fluxes.map((_, i) =>
    fluxes.map((_, j) => {
      if (i === j) return 1.0;
      const a = Math.abs(fluxes[i]), b = Math.abs(fluxes[j]);
      if (a < 1e-4 || b < 1e-4) return 0;
      return (2 * a * b) / (a * a + b * b);
    })
  );
}

function atpFlux(atp, phase, wGln, wO2) {
  const mm  = (a, km) => a / (a + km);
  const mmI = (a, km) => km / (a + km);
  const gly  = 2.0  * (mmI(atp, 0.30) + 0.12) * (1 - phase * 0.65);
  const oxph = 29.0 *  mm(atp, 0.58) * wO2  * (1 - phase * 0.48);
  const gln  = 4.5  *  mm(atp, 0.40) * wGln * (1 - phase * 0.25);
  return { total: gly + oxph + gln, gly, oxph, gln };
}

// ─── Culture phases ───────────────────────────────────────────────────────────

const PHASES = [
  { key: "exp",  label: "Exponential", sub: "High μ · glucose driven · aerobic",    color: "#22c55e", val: 0   },
  { key: "stat", label: "Stationary",  sub: "Nutrient depletion · metabolic arrest",color: "#f59e0b", val: 0.5 },
  { key: "dec",  label: "Decline",     sub: "Lactate accumulation · cell death",    color: "#B63E96", val: 1   },
];

// ─── Component ────────────────────────────────────────────────────────────────

export default function Bioreactor() {
  const holoRef  = useRef(null);
  const fluxRef  = useRef(null);
  const glHolo   = useRef(null);
  const glFlux   = useRef(null);
  const animRef  = useRef(null);
  const startRef = useRef(null);

  const [phaseKey, setPhaseKey]     = useState("exp");
  const [phaseVal, setPhaseVal]     = useState(0);
  const [atp, setAtp]               = useState(0.4);    // 0–1 → 0.5–15 mM
  const [wGln, setWGln]             = useState(0.7);
  const [wO2, setWO2]               = useState(0.8);
  const [entropy, setEntropy]       = useState({ Sk: 0, St: 0, Se: 0 });
  const [coupling, setCoupling]     = useState(null);
  const [transitioning, setTransitioning] = useState(false);

  const phaseRef       = useRef(0);
  const targetPhaseRef = useRef(0);

  const switchPhase = useCallback((key) => {
    const p = PHASES.find((x) => x.key === key);
    if (!p) return;
    targetPhaseRef.current = p.val;
    setPhaseKey(key);
    setTransitioning(true);
  }, []);

  // Init WebGL
  useEffect(() => {
    const h = holoRef.current;
    const f = fluxRef.current;
    if (!h || !f) return;

    const rszH = () => {
      h.width = h.offsetWidth; h.height = h.offsetHeight;
      glHolo.current?.gl.viewport(0, 0, h.width, h.height);
    };
    const rszF = () => {
      f.width = f.offsetWidth; f.height = f.offsetHeight;
      glFlux.current?.gl.viewport(0, 0, f.width, f.height);
    };
    rszH(); rszF();
    window.addEventListener("resize", rszH);
    window.addEventListener("resize", rszF);

    glHolo.current = makeGL(h, FRAG_HOLO);
    glFlux.current = makeGL(f, FRAG_FLUX);
    startRef.current = performance.now();

    return () => {
      window.removeEventListener("resize", rszH);
      window.removeEventListener("resize", rszF);
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  // Stale-closure guards for animation loop
  const atpRef  = useRef(atp);
  const wGlnRef = useRef(wGln);
  const wO2Ref  = useRef(wO2);
  useEffect(() => { atpRef.current  = atp;  }, [atp]);
  useEffect(() => { wGlnRef.current = wGln; }, [wGln]);
  useEffect(() => { wO2Ref.current  = wO2;  }, [wO2]);

  // Animation loop
  useEffect(() => {
    const loop = () => {
      animRef.current = requestAnimationFrame(loop);
      const t  = (performance.now() - (startRef.current || 0)) * 0.001;
      const a  = atpRef.current;
      const wg = wGlnRef.current;
      const wo = wO2Ref.current;

      const curr = phaseRef.current;
      const tgt  = targetPhaseRef.current;
      if (Math.abs(curr - tgt) > 0.005) {
        phaseRef.current = curr + (tgt - curr) * 0.055;
        setPhaseVal(phaseRef.current);
        setEntropy(computeSEntropy(a, phaseRef.current, wg, wo));
        setCoupling(computeCouplingMatrix(a, phaseRef.current, wg, wo));
      } else if (transitioning) {
        phaseRef.current = tgt;
        setPhaseVal(tgt);
        setTransitioning(false);
      }

      const ph = phaseRef.current;
      if (glHolo.current) {
        const { gl, u } = glHolo.current;
        gl.uniform1f(u("u_time"),  t);
        gl.uniform1f(u("u_atp"),   a);
        gl.uniform1f(u("u_phase"), ph);
        gl.uniform1f(u("u_w_gln"), wg);
        gl.uniform1f(u("u_w_O2"),  wo);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
      if (glFlux.current) {
        const { gl, u } = glFlux.current;
        gl.uniform1f(u("u_atp"),   a);
        gl.uniform1f(u("u_phase"), ph);
        gl.uniform1f(u("u_w_gln"), wg);
        gl.uniform1f(u("u_w_O2"),  wo);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
    };
    setEntropy(computeSEntropy(atp, 0, wGln, wO2));
    setCoupling(computeCouplingMatrix(atp, 0, wGln, wO2));
    loop();
    return () => cancelAnimationFrame(animRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Recompute when sliders change
  useEffect(() => {
    setEntropy(computeSEntropy(atp, phaseRef.current, wGln, wO2));
    setCoupling(computeCouplingMatrix(atp, phaseRef.current, wGln, wO2));
  }, [atp, wGln, wO2]);

  const atpMM    = (0.5 + atp * 14.5).toFixed(1);
  const flux     = atpFlux(atp, phaseVal, wGln, wO2);
  const partN    = Math.max(1, Math.round(atp * 12 + 1));
  const curPhase = PHASES.find((p) => p.key === phaseKey);

  return (
    <>
      <Head>
        <title>Bioreactor Instrument — Categorical Spectrometry</title>
      </Head>

      <div className="w-full min-h-screen bg-dark text-light font-mont flex flex-col">
        {/* ── Header ── */}
        <header className="flex items-center justify-between px-8 py-5 border-b border-white/10">
          <Link
            href="/"
            className="text-xs tracking-widest uppercase opacity-40 hover:opacity-80 transition-opacity"
          >
            ← Back
          </Link>
          <div className="text-center">
            <h1 className="text-sm tracking-widest uppercase font-medium opacity-80">
              Cellular Metabolic Instrument
            </h1>
            <p className="text-xs mt-0.5 opacity-30 tracking-wider">
              Bioreactor · ATP-Constrained Dynamics · S-Space Navigation
            </p>
          </div>
          <div className="text-xs tracking-widest uppercase opacity-40 hover:opacity-80 transition-opacity">
            <Link href="/polymorphism">← Polymorphism</Link>
          </div>
        </header>

        {/* ── Main layout ── */}
        <div className="flex flex-1 overflow-hidden" style={{ minHeight: "calc(100vh - 74px)" }}>

          {/* ── Hologram canvas ── */}
          <div className="flex-1 relative" style={{ minWidth: 0 }}>
            <canvas
              ref={holoRef}
              className="w-full h-full block"
              style={{ background: "#030705" }}
            />

            {/* Metabolite labels along X */}
            <div
              className="absolute bottom-3 left-0 right-0 pointer-events-none"
              style={{ fontSize: "0.58rem", opacity: 0.38, letterSpacing: "0.1em" }}
            >
              <div style={{ position: "absolute", left: "8%"  }}>Glucose</div>
              <div style={{ position: "absolute", left: "25%" }}>Lactate</div>
              <div style={{ position: "absolute", left: "47%" }}>Glutamine</div>
              <div style={{ position: "absolute", left: "68%" }}>O₂</div>
              <div style={{ position: "absolute", left: "83%" }}>ATP/ADP</div>
            </div>
            <div
              className="absolute bottom-8 left-0 right-0 text-center pointer-events-none"
              style={{ fontSize: "0.58rem", opacity: 0.22, letterSpacing: "0.12em" }}
            >
              metabolic oscillation frequency →
            </div>

            {/* Phase axis */}
            <div
              className="absolute left-2 top-0 bottom-10 flex flex-col justify-between pointer-events-none"
              style={{ fontSize: "0.58rem", opacity: 0.35, letterSpacing: "0.1em" }}
            >
              <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>2π</span>
              <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>ATP phase φ</span>
              <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>0</span>
            </div>

            {/* Legend */}
            <div
              className="absolute top-3 right-4 flex flex-col gap-1 pointer-events-none"
              style={{ fontSize: "0.6rem", opacity: 0.72 }}
            >
              <span style={{ color: "#22c55e" }}>▬ Glucose (glycolytic)</span>
              <span style={{ color: "#ef4444" }}>▬ Lactate (byproduct)</span>
              <span style={{ color: "#60a5fa" }}>▬ Glutamine (TCA)</span>
              <span style={{ color: "#e2e8f0" }}>▬ O₂ (oxphos)</span>
            </div>
          </div>

          {/* ── Right panel ── */}
          <aside
            className="flex flex-col"
            style={{
              width: "390px",
              flexShrink: 0,
              borderLeft: "1px solid rgba(255,255,255,0.08)",
              background: "rgba(255,255,255,0.02)",
            }}
          >
            {/* ATP flux canvas */}
            <div
              style={{
                height: "clamp(100px, 22vh, 175px)",
                flexShrink: 0,
                borderBottom: "1px solid rgba(255,255,255,0.07)",
              }}
            >
              <p
                className="px-5 pt-3 pb-1"
                style={{
                  fontSize: "0.6rem",
                  letterSpacing: "0.12em",
                  opacity: 0.35,
                  textTransform: "uppercase",
                }}
              >
                ATP Production Spectrum d[ATP]/dt
              </p>
              <canvas
                ref={fluxRef}
                className="w-full block"
                style={{ height: "calc(100% - 28px)", background: "#030a05" }}
              />
            </div>

            {/* Scrollable controls */}
            <div className="flex flex-col gap-5 px-7 py-5 overflow-y-auto flex-1">

              {/* Culture phase */}
              <section>
                <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                  Culture Phase
                </p>
                <div className="flex flex-col gap-2">
                  {PHASES.map(({ key, label, sub, color }) => (
                    <button
                      key={key}
                      onClick={() => switchPhase(key)}
                      className="py-2 px-3 rounded text-left transition-all"
                      style={{
                        background: phaseKey === key ? `${color}22` : "rgba(255,255,255,0.04)",
                        border: phaseKey === key
                          ? `1px solid ${color}88`
                          : "1px solid rgba(255,255,255,0.08)",
                      }}
                    >
                      <span
                        className="block"
                        style={{
                          fontSize: "0.72rem",
                          color: phaseKey === key ? color : "rgba(255,255,255,0.55)",
                        }}
                      >
                        {label}
                      </span>
                      <span className="block mt-0.5" style={{ fontSize: "0.55rem", opacity: 0.35 }}>
                        {sub}
                      </span>
                    </button>
                  ))}
                </div>
              </section>

              {/* ATP slider */}
              <section>
                <div className="flex justify-between mb-2">
                  <p className="text-xs tracking-widest uppercase opacity-40">ATP Concentration</p>
                  <span style={{ color: "#f59e0b", fontSize: "0.72rem" }}>{atpMM} mM</span>
                </div>
                <input
                  type="range" min={0} max={1} step={0.01} value={atp}
                  onChange={(e) => setAtp(parseFloat(e.target.value))}
                  className="w-full"
                  style={{ accentColor: "#f59e0b" }}
                />
                <div className="flex justify-between mt-1" style={{ fontSize: "0.55rem", opacity: 0.28 }}>
                  <span>0.5 mM (anaerobic)</span>
                  <span>15 mM (aerobic)</span>
                </div>
              </section>

              {/* S-entropy */}
              <section>
                <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                  S-Entropy Coordinates
                </p>
                {[
                  { label: "S_k  Metabolic", val: entropy.Sk, desc: "Shannon entropy of flux distribution", color: "#22c55e" },
                  { label: "S_t  Temporal",  val: entropy.St, desc: "log(τ_gen/τ_ATP) / log(τ_gen/τ_P)",  color: "#f59e0b" },
                  { label: "S_e  Evolution", val: entropy.Se, desc: "Metabolic coupling edge density",     color: "#B63E96" },
                ].map(({ label, val, desc, color }) => (
                  <div key={label} className="mb-3">
                    <div className="flex justify-between mb-1" style={{ fontSize: "0.65rem" }}>
                      <span className="opacity-60 tracking-wider">{label}</span>
                      <span style={{ color, fontVariantNumeric: "tabular-nums" }}>{val.toFixed(4)}</span>
                    </div>
                    <div
                      className="w-full rounded-full overflow-hidden"
                      style={{ height: 3, background: "rgba(255,255,255,0.08)" }}
                    >
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{ width: `${val * 100}%`, background: color }}
                      />
                    </div>
                    <p style={{ fontSize: "0.55rem", opacity: 0.28, marginTop: 3 }}>{desc}</p>
                  </div>
                ))}
              </section>

              {/* ATP flux breakdown */}
              <section>
                <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                  ATP Flux Breakdown
                </p>
                <div
                  className="rounded p-3"
                  style={{
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.07)",
                    fontSize: "0.62rem",
                  }}
                >
                  {[
                    ["Glycolytic",      flux.gly,   "#22c55e"],
                    ["OXPHOS",          flux.oxph,  "#e2e8f0"],
                    ["Glutaminolysis",  flux.gln,   "#60a5fa"],
                    ["Total d[ATP]/dt", flux.total, "#f59e0b"],
                  ].map(([label, val, color]) => (
                    <div key={label} className="flex justify-between mb-1.5">
                      <span className="opacity-40">{label}</span>
                      <span style={{ color }}>{val.toFixed(2)} mmol/h/g</span>
                    </div>
                  ))}
                  <div
                    className="mt-2 pt-2"
                    style={{ borderTop: "1px solid rgba(255,255,255,0.07)", opacity: 0.3, fontSize: "0.57rem" }}
                  >
                    dx/d[ATP] — ATP is the time coordinate
                  </div>
                </div>
              </section>

              {/* Substrate weights */}
              <section>
                <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                  Substrate Weights
                </p>
                {[
                  { label: "w_Gln  Glutamine", val: wGln, set: setWGln, color: "#60a5fa" },
                  { label: "w_O₂   Dissolved",  val: wO2,  set: setWO2,  color: "#e2e8f0" },
                ].map(({ label, val, set, color }) => (
                  <div key={label} className="mb-3">
                    <div className="flex justify-between mb-1.5" style={{ fontSize: "0.65rem" }}>
                      <span className="opacity-60">{label}</span>
                      <span style={{ color }}>{val.toFixed(2)}</span>
                    </div>
                    <input
                      type="range" min={0} max={1} step={0.01} value={val}
                      onChange={(e) => set(parseFloat(e.target.value))}
                      className="w-full"
                      style={{ accentColor: color }}
                    />
                  </div>
                ))}
              </section>

              {/* Metabolic coupling matrix */}
              {coupling && (
                <section>
                  <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                    Metabolic Coupling Matrix K̃ᵢⱼ
                  </p>
                  <div style={{ fontSize: "0.54rem" }}>
                    <table className="w-full" style={{ borderCollapse: "collapse" }}>
                      <thead>
                        <tr>
                          <td />
                          {MET_SHORT.map((m) => (
                            <td key={m} className="text-center pb-1" style={{ opacity: 0.4 }}>
                              {m}
                            </td>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {coupling.map((row, i) => (
                          <tr key={i}>
                            <td className="pr-1 opacity-40">{MET_SHORT[i]}</td>
                            {row.map((v, j) => (
                              <td key={j} className="p-px text-center">
                                <div
                                  title={v.toFixed(3)}
                                  style={{
                                    width: 30,
                                    height: 14,
                                    borderRadius: 2,
                                    background: `rgba(${
                                      v > 0.8
                                        ? "34,197,94"
                                        : v > 0.5
                                        ? "245,158,11"
                                        : "255,255,255"
                                    }, ${v * 0.85})`,
                                  }}
                                />
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="mt-1.5 opacity-25">
                      Green = strongly coupled (K̃&gt;0.8). Changes with [ATP] and growth phase.
                    </p>
                  </div>
                </section>
              )}

              {/* Categorical identification */}
              <section>
                <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                  Categorical Identification
                </p>
                <div
                  className="rounded p-3 text-xs"
                  style={{
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    fontFamily: "monospace",
                  }}
                >
                  {[
                    ["Partition depth n",  partN,                            "#22c55e"],
                    ["C(n) = 2n²",         2 * partN * partN,                "#f59e0b"],
                    ["Phase address",      `${phaseKey}·${atpMM}mM`,         "#60a5fa"],
                    ["d_partition(X,Y)",   "= d_CV(I_X, I_Y)",               "#B63E96"],
                  ].map(([label, val, color]) => (
                    <div key={label} className="flex justify-between mb-1.5">
                      <span className="opacity-40">{label}</span>
                      <span style={{ color }}>{val}</span>
                    </div>
                  ))}
                  <div
                    className="mt-2 pt-2"
                    style={{ borderTop: "1px solid rgba(255,255,255,0.07)" }}
                  >
                    <div className="flex justify-between mb-1">
                      <span className="opacity-40">Observer model</span>
                      <span style={{ color: "#f5f5f5", opacity: 0.7 }}>99% membrane</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="opacity-40">Fallback</span>
                      <span style={{ color: "#f5f5f5", opacity: 0.7 }}>1% DNA</span>
                    </div>
                  </div>
                </div>
                <p className="text-xs mt-2 opacity-25 leading-relaxed">
                  Membrane quantum observer: capacitance as bounded oscillator.
                  GPU interference pattern IS the physical measurement. Zero free parameters.
                </p>
              </section>

            </div>
          </aside>
        </div>
      </div>
    </>
  );
}
