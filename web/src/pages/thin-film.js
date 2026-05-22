import Head from "next/head";
import Link from "next/link";
import { useEffect, useRef, useState, useCallback } from "react";

// ─── Thin-film interference fragment shader ───────────────────────────────────
// Computes I(λ) = Fresnel reflectance for a single dielectric layer on Si.
// This IS the GPU-Interference Isomorphism: the fragment shader executing
// I_int = (I_A + I_B + 2√(I_A·I_B)·cos(Δφ))/2 is identical to the physical
// optical measurement.  No simulation — same bounded-oscillator math.

const VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main(){
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// Main canvas: renders a full-width thin-film "wafer" view
// X = position across wafer (maps to thickness variation for visualization)
// Y = wavelength within each column
const FRAG_FILM = `
precision highp float;
varying vec2 v_uv;
uniform float u_d;       // film thickness nm
uniform float u_n1;      // film refractive index
uniform float u_n0;      // ambient (air = 1.0)
uniform float u_n2;      // substrate (Si ~3.88)
uniform float u_time;

// Convert wavelength (nm) to linear sRGB
vec3 wavelengthToRGB(float lam){
  float r=0., g=0., b=0.;
  if(lam>= 380. && lam< 440.){ r=(440.-lam)/60.; b=1.; }
  else if(lam>=440. && lam< 490.){ g=(lam-440.)/50.; b=1.; }
  else if(lam>=490. && lam< 510.){ g=1.; b=(510.-lam)/20.; }
  else if(lam>=510. && lam< 580.){ r=(lam-510.)/70.; g=1.; }
  else if(lam>=580. && lam< 645.){ r=1.; g=(645.-lam)/65.; }
  else if(lam>=645. && lam<=780.){ r=1.; }
  float fac=1.;
  if(lam>=380.&&lam<420.)      fac=0.3+0.7*(lam-380.)/40.;
  else if(lam>700.&&lam<=780.) fac=0.3+0.7*(780.-lam)/80.;
  return vec3(r,g,b)*fac;
}

// Fresnel amplitude coefficient at interface n_a → n_b (normal incidence)
float r_amp(float na, float nb){ return (na - nb)/(na + nb); }

// Thin-film reflectance via interference formula
float thinFilmR(float lam, float d, float n0, float n1, float n2){
  float r01 = r_amp(n0, n1);
  float r12 = r_amp(n1, n2);
  float delta = 2.0 * 3.14159265 * n1 * d / lam; // phase round-trip
  // Intensity reflectance (Fabry-Perot / two-beam interference)
  float I0 = r01*r01;
  float I1 = r12*r12;
  float I_int = (I0 + I1 + 2.0*sqrt(I0*I1)*cos(2.0*delta)) /
                (1.0 + I0*I1 + 2.0*sqrt(I0*I1)*cos(2.0*delta));
  return clamp(I_int, 0.0, 1.0);
}

void main(){
  // x = position on wafer (slight thickness gradient for visual interest)
  // y = wavelength mapping (bottom=380nm, top=780nm)
  float lam = mix(380., 780., v_uv.y);

  // Subtle thickness variation across wafer for visual interest
  float dLocal = u_d * (1.0 + 0.04 * sin(v_uv.x * 6.28 + u_time * 0.3));

  float R = thinFilmR(lam, dLocal, u_n0, u_n1, u_n2);

  // Reflected color = R * (wavelength color) + (1-R) * dark substrate
  vec3 col = wavelengthToRGB(lam) * R;
  // Add faint substrate contribution
  vec3 substrate = vec3(0.06, 0.06, 0.08);
  col = col + substrate * (1.0 - R) * 0.25;

  // Vignetted edges
  float vx = smoothstep(0.0,0.08,v_uv.x)*smoothstep(1.0,0.92,v_uv.x);
  float vy = smoothstep(0.0,0.04,v_uv.y)*smoothstep(1.0,0.96,v_uv.y);
  col *= vx * vy;

  gl_FragColor = vec4(col, 1.0);
}`;

// Spectrum canvas: 1D reflectance vs wavelength plot
const FRAG_SPECTRUM = `
precision highp float;
varying vec2 v_uv;
uniform float u_d;
uniform float u_n1;
uniform float u_n0;
uniform float u_n2;

float r_amp(float na, float nb){ return (na - nb)/(na + nb); }

float thinFilmR(float lam, float d, float n0, float n1, float n2){
  float r01 = r_amp(n0, n1);
  float r12 = r_amp(n1, n2);
  float delta = 2.0 * 3.14159265 * n1 * d / lam;
  float I0 = r01*r01;
  float I1 = r12*r12;
  return clamp((I0+I1+2.0*sqrt(I0*I1)*cos(2.0*delta))/
               (1.0+I0*I1+2.0*sqrt(I0*I1)*cos(2.0*delta)), 0., 1.);
}

vec3 wavelengthToRGB(float lam){
  float r=0.,g=0.,b=0.;
  if(lam>=380.&&lam<440.){r=(440.-lam)/60.;b=1.;}
  else if(lam>=440.&&lam<490.){g=(lam-440.)/50.;b=1.;}
  else if(lam>=490.&&lam<510.){g=1.;b=(510.-lam)/20.;}
  else if(lam>=510.&&lam<580.){r=(lam-510.)/70.;g=1.;}
  else if(lam>=580.&&lam<645.){r=1.;g=(645.-lam)/65.;}
  else if(lam>=645.&&lam<=780.){r=1.;}
  float fac=1.;
  if(lam>=380.&&lam<420.)      fac=0.3+0.7*(lam-380.)/40.;
  else if(lam>700.&&lam<=780.) fac=0.3+0.7*(780.-lam)/80.;
  return vec3(r,g,b)*fac;
}

void main(){
  float lam = mix(380., 780., v_uv.x);
  float R   = thinFilmR(lam, u_d, u_n0, u_n1, u_n2);

  // Background grid
  float gridV = mod(v_uv.y * 4.0, 1.0);
  vec3 bg = vec3(0.10, 0.10, 0.15) + 0.015 * step(0.98, gridV);

  // Spectrum curve
  float lineH = 0.1 + R * 0.8;
  float onLine = smoothstep(0.012, 0.0, abs(v_uv.y - lineH));
  float fill   = step(v_uv.y, lineH) * 0.35;

  vec3 specCol = wavelengthToRGB(lam);
  vec3 col = bg + (specCol * (onLine + fill));

  // Axis
  float axisV = smoothstep(0.007, 0.0, abs(v_uv.y - 0.1));
  col += vec3(0.3,0.3,0.3) * axisV;

  gl_FragColor = vec4(clamp(col,0.,1.), 1.0);
}`;

// ─── WebGL helpers ─────────────────────────────────────────────────────────

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

// ─── S-entropy for thin-film ──────────────────────────────────────────────

function computeSEntropy(d, n1) {
  // Sample reflected spectrum at 32 wavelengths → Shannon entropy
  const lambdas = Array.from({length: 32}, (_, i) => 380 + i * (400 / 31));
  const r_amp = (na, nb) => (na - nb) / (na + nb);
  const R = lambdas.map((lam) => {
    const r01 = r_amp(1.0, n1), r12 = r_amp(n1, 3.88);
    const delta = 2 * Math.PI * n1 * d / lam;
    const I0 = r01 * r01, I1 = r12 * r12;
    return Math.min(1, Math.max(0,
      (I0 + I1 + 2 * Math.sqrt(I0 * I1) * Math.cos(2 * delta)) /
      (1 + I0 * I1 + 2 * Math.sqrt(I0 * I1) * Math.cos(2 * delta))
    ));
  });
  const total = R.reduce((a, b) => a + b, 0) || 1;
  const pn = R.map((v) => v / total);
  const H = -pn.reduce((s, p) => s + (p > 1e-10 ? p * Math.log(p) : 0), 0);
  const Sk = Math.min(1, H / Math.log(32));

  // St: temporal entropy — thickness relative to coherence length
  const lc = 550 * 550 / (400 * n1); // coherence length in nm
  const St = Math.min(1, Math.log(1 + d / lc) / Math.log(1 + 500 / lc));

  // Se: coupling between optical modes (Fabry-Perot resonances)
  // Count distinct interference maxima → normalized edge density proxy
  let peaks = 0;
  for (let i = 1; i < R.length - 1; i++)
    if (R[i] > R[i-1] && R[i] > R[i+1] && R[i] > 0.05) peaks++;
  const maxPeaks = Math.max(1, Math.floor(2 * n1 * d / 400));
  const Se = Math.min(1, peaks / maxPeaks);

  return { Sk, St, Se };
}

// Optical mode "partition depth" n from mode count
function partitionDepth(d, n1) {
  return Math.max(1, Math.round(2 * n1 * d / 550));
}

// ─── Materials ───────────────────────────────────────────────────────────────

const MATERIALS = {
  SiO2:  { label: "SiO₂",  n: 1.46, desc: "Thermal oxide / gate dielectric" },
  Si3N4: { label: "Si₃N₄", n: 2.00, desc: "Nitride passivation / hardmask" },
  HfO2:  { label: "HfO₂",  n: 2.09, desc: "High-κ gate dielectric (FinFET)" },
  Al2O3: { label: "Al₂O₃", n: 1.63, desc: "ALD dielectric / passivation" },
};

// ─── Component ───────────────────────────────────────────────────────────────

export default function ThinFilm() {
  const filmRef = useRef(null);
  const specRef = useRef(null);
  const glFilm = useRef(null);
  const glSpec = useRef(null);
  const animRef = useRef(null);
  const startRef = useRef(null);

  const [thickness, setThickness] = useState(120);
  const [material, setMaterial] = useState("SiO2");
  const [entropy, setEntropy] = useState({ Sk: 0, St: 0, Se: 0 });
  const [pDepth, setPDepth] = useState(1);

  const n1 = MATERIALS[material].n;
  const n0 = 1.0;
  const n2 = 3.88;

  // Re-compute analytics when params change
  useEffect(() => {
    setEntropy(computeSEntropy(thickness, n1));
    setPDepth(partitionDepth(thickness, n1));
  }, [thickness, n1]);

  // Init WebGL for both canvases
  useEffect(() => {
    const filmCanvas = filmRef.current;
    const specCanvas = specRef.current;
    if (!filmCanvas || !specCanvas) return;

    const resizeFilm = () => {
      filmCanvas.width = filmCanvas.offsetWidth;
      filmCanvas.height = filmCanvas.offsetHeight;
      glFilm.current?.gl.viewport(0, 0, filmCanvas.width, filmCanvas.height);
    };
    const resizeSpec = () => {
      specCanvas.width = specCanvas.offsetWidth;
      specCanvas.height = specCanvas.offsetHeight;
      glSpec.current?.gl.viewport(0, 0, specCanvas.width, specCanvas.height);
    };
    resizeFilm();
    resizeSpec();
    window.addEventListener("resize", resizeFilm);
    window.addEventListener("resize", resizeSpec);

    glFilm.current = makeGL(filmCanvas, FRAG_FILM);
    glSpec.current = makeGL(specCanvas, FRAG_SPECTRUM);
    startRef.current = performance.now();

    return () => {
      window.removeEventListener("resize", resizeFilm);
      window.removeEventListener("resize", resizeSpec);
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  // Animation loop — reads current slider values via closure refs
  const thicknessRef = useRef(thickness);
  const n1Ref = useRef(n1);
  useEffect(() => { thicknessRef.current = thickness; }, [thickness]);
  useEffect(() => { n1Ref.current = n1; }, [n1]);

  useEffect(() => {
    const loop = () => {
      animRef.current = requestAnimationFrame(loop);
      const t = (performance.now() - (startRef.current || 0)) * 0.001;
      const d = thicknessRef.current;
      const n = n1Ref.current;

      if (glFilm.current) {
        const { gl, u } = glFilm.current;
        gl.uniform1f(u("u_d"), d);
        gl.uniform1f(u("u_n1"), n);
        gl.uniform1f(u("u_n0"), n0);
        gl.uniform1f(u("u_n2"), n2);
        gl.uniform1f(u("u_time"), t);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
      if (glSpec.current) {
        const { gl, u } = glSpec.current;
        gl.uniform1f(u("u_d"), d);
        gl.uniform1f(u("u_n1"), n);
        gl.uniform1f(u("u_n0"), n0);
        gl.uniform1f(u("u_n2"), n2);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
    };
    loop();
    return () => cancelAnimationFrame(animRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <Head>
        <title>Thin-Film Instrument — Categorical Spectrometry</title>
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
              Thin-Film Optical Instrument
            </h1>
            <p className="text-xs mt-0.5 opacity-30 tracking-wider">
              Semiconductor Fab · Fresnel Interference · Partition Coordinates
            </p>
          </div>
          <div className="text-xs tracking-widest uppercase opacity-40 hover:opacity-80 transition-opacity">
            <Link href="/polymorphism">← Polymorphism</Link>
          </div>
        </header>

        {/* ── Layout: film view (top) + controls/spectrum (bottom) ── */}
        <div className="flex flex-1 flex-col overflow-hidden">

          {/* Film canvas — takes top ~55% */}
          <div
            className="relative w-full"
            style={{ height: "clamp(220px, 45vh, 400px)", flexShrink: 0 }}
          >
            <canvas
              ref={filmRef}
              className="w-full h-full block"
              style={{ background: "#0b0b10" }}
            />
            {/* Overlay: thickness annotation */}
            <div
              className="absolute bottom-3 left-0 right-0 flex justify-center pointer-events-none"
              style={{ fontSize: "0.62rem", opacity: 0.4, letterSpacing: "0.12em" }}
            >
              <span>
                {MATERIALS[material].label} · d = {thickness} nm · n = {n1.toFixed(2)} · n_Si = 3.88
              </span>
            </div>
            <div
              className="absolute top-3 left-4 pointer-events-none"
              style={{ fontSize: "0.6rem", opacity: 0.3, letterSpacing: "0.1em" }}
            >
              <div>λ = 780 nm</div>
            </div>
            <div
              className="absolute bottom-8 left-4 pointer-events-none"
              style={{ fontSize: "0.6rem", opacity: 0.3, letterSpacing: "0.1em" }}
            >
              <div>λ = 380 nm</div>
            </div>
          </div>

          {/* Bottom row */}
          <div className="flex flex-1" style={{ borderTop: "1px solid rgba(255,255,255,0.07)" }}>

            {/* Reflected spectrum plot */}
            <div
              className="flex flex-col"
              style={{
                width: "38%",
                borderRight: "1px solid rgba(255,255,255,0.07)",
                flexShrink: 0,
              }}
            >
              <p
                className="px-5 pt-4 pb-1"
                style={{ fontSize: "0.6rem", letterSpacing: "0.12em", opacity: 0.35, textTransform: "uppercase" }}
              >
                Reflected spectrum R(λ)
              </p>
              <canvas
                ref={specRef}
                className="w-full flex-1 block"
                style={{ background: "#0d0d16" }}
              />
              <div
                className="flex justify-between px-5 pb-2"
                style={{ fontSize: "0.55rem", opacity: 0.28, letterSpacing: "0.08em" }}
              >
                <span>380 nm</span>
                <span>Wavelength</span>
                <span>780 nm</span>
              </div>
            </div>

            {/* Controls + readouts */}
            <div
              className="flex flex-1 gap-0"
              style={{ minWidth: 0, overflowX: "hidden" }}
            >
              {/* Controls */}
              <div
                className="flex flex-col gap-5 px-7 py-5 overflow-y-auto"
                style={{
                  width: "50%",
                  borderRight: "1px solid rgba(255,255,255,0.06)",
                  flexShrink: 0,
                }}
              >
                {/* Material selector */}
                <section>
                  <p className="text-xs tracking-widest uppercase opacity-40 mb-3">Material</p>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(MATERIALS).map(([key, { label, desc }]) => (
                      <button
                        key={key}
                        onClick={() => setMaterial(key)}
                        className="py-2 px-3 rounded text-left transition-all"
                        style={{
                          background:
                            material === key
                              ? "rgba(88,230,217,0.14)"
                              : "rgba(255,255,255,0.04)",
                          border:
                            material === key
                              ? "1px solid rgba(88,230,217,0.45)"
                              : "1px solid rgba(255,255,255,0.08)",
                        }}
                      >
                        <span
                          className="block"
                          style={{
                            fontSize: "0.72rem",
                            color: material === key ? "#58E6D9" : "rgba(255,255,255,0.6)",
                          }}
                        >
                          {label}
                        </span>
                        <span
                          className="block mt-0.5"
                          style={{ fontSize: "0.55rem", opacity: 0.35 }}
                        >
                          n={MATERIALS[key].n.toFixed(2)}
                        </span>
                      </button>
                    ))}
                  </div>
                  <p className="mt-2" style={{ fontSize: "0.58rem", opacity: 0.3 }}>
                    {MATERIALS[material].desc}
                  </p>
                </section>

                {/* Thickness slider */}
                <section>
                  <div className="flex justify-between mb-2">
                    <p className="text-xs tracking-widest uppercase opacity-40">
                      Film Thickness
                    </p>
                    <span style={{ color: "#58E6D9", fontSize: "0.72rem" }}>
                      {thickness} nm
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={600}
                    step={1}
                    value={thickness}
                    onChange={(e) => setThickness(parseInt(e.target.value))}
                    className="w-full"
                    style={{ accentColor: "#58E6D9" }}
                  />
                  <div
                    className="flex justify-between mt-1"
                    style={{ fontSize: "0.55rem", opacity: 0.28 }}
                  >
                    <span>0 nm</span>
                    <span>600 nm</span>
                  </div>
                </section>

                {/* Interference equation */}
                <section>
                  <p className="text-xs tracking-widest uppercase opacity-40 mb-2">
                    Interference Identity
                  </p>
                  <div
                    className="rounded p-3 text-xs leading-relaxed"
                    style={{
                      background: "rgba(255,255,255,0.03)",
                      border: "1px solid rgba(255,255,255,0.07)",
                      fontFamily: "monospace",
                      fontSize: "0.62rem",
                    }}
                  >
                    <div style={{ color: "#58E6D9", marginBottom: 4 }}>
                      I = (I₀ + I₁ + 2√(I₀I₁)cos(Δφ)) /
                    </div>
                    <div style={{ color: "#58E6D9", marginBottom: 6 }}>
                      &nbsp;&nbsp;&nbsp;&nbsp;(1 + I₀I₁ + 2√(I₀I₁)cos(Δφ))
                    </div>
                    <div style={{ opacity: 0.4 }}>
                      Δφ = 4πn₁d/λ &nbsp;·&nbsp; r₀₁={(((n0 - n1) / (n0 + n1)).toFixed(3))} &nbsp;·&nbsp; r₁₂={((n1 - n2) / (n1 + n2)).toFixed(3)}
                    </div>
                    <p className="mt-2" style={{ opacity: 0.3, fontFamily: "inherit" }}>
                      Fragment shader executing this formula IS the physical
                      ellipsometric measurement — GPU-Interference Isomorphism.
                    </p>
                  </div>
                </section>
              </div>

              {/* Readouts */}
              <div
                className="flex flex-col gap-5 px-7 py-5 overflow-y-auto"
                style={{ flex: 1, minWidth: 0 }}
              >
                {/* S-entropy */}
                <section>
                  <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                    S-Entropy Coordinates
                  </p>
                  {[
                    { label: "S_k  Spectral", val: entropy.Sk, desc: "Shannon entropy of R(λ)", color: "#58E6D9" },
                    { label: "S_t  Temporal", val: entropy.St, desc: "d / coherence length ratio", color: "#B63E96" },
                    { label: "S_e  Evolution", val: entropy.Se, desc: "Interference peak density", color: "#9070d8" },
                  ].map(({ label, val, desc, color }) => (
                    <div key={label} className="mb-3">
                      <div
                        className="flex justify-between mb-1"
                        style={{ fontSize: "0.65rem" }}
                      >
                        <span className="opacity-60 tracking-wider">{label}</span>
                        <span style={{ color, fontVariantNumeric: "tabular-nums" }}>
                          {val.toFixed(4)}
                        </span>
                      </div>
                      <div
                        className="w-full rounded-full overflow-hidden"
                        style={{ height: 3, background: "rgba(255,255,255,0.08)" }}
                      >
                        <div
                          className="h-full rounded-full transition-all duration-200"
                          style={{ width: `${val * 100}%`, background: color }}
                        />
                      </div>
                      <p style={{ fontSize: "0.55rem", opacity: 0.28, marginTop: 2 }}>
                        {desc}
                      </p>
                    </div>
                  ))}
                </section>

                {/* Partition coordinates */}
                <section>
                  <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                    Partition Coordinates
                  </p>
                  <div
                    className="rounded p-3"
                    style={{
                      background: "rgba(255,255,255,0.03)",
                      border: "1px solid rgba(255,255,255,0.07)",
                      fontSize: "0.63rem",
                      fontFamily: "monospace",
                    }}
                  >
                    {[
                      ["n (shell depth)", pDepth, "#58E6D9"],
                      ["l (angular mode)", Math.min(pDepth - 1, 1), "#B63E96"],
                      ["m (orientation)", 0, "#9070d8"],
                      ["s (chirality)", "+½", "#88aaff"],
                    ].map(([label, val, color]) => (
                      <div key={label} className="flex justify-between mb-1.5">
                        <span className="opacity-40">{label}</span>
                        <span style={{ color }}>{val}</span>
                      </div>
                    ))}
                    <div
                      className="flex justify-between mt-2 pt-2"
                      style={{ borderTop: "1px solid rgba(255,255,255,0.07)" }}
                    >
                      <span className="opacity-40">C(n) = 2n²</span>
                      <span style={{ color: "#58E6D9" }}>{2 * pDepth * pDepth} states</span>
                    </div>
                  </div>
                </section>

                {/* Process state */}
                <section>
                  <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                    Process State
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
                      ["Fab process", thickness < 10 ? "Bare Si" : thickness < 50 ? "Native oxide" : thickness < 200 ? "Thin gate" : "Field oxide"],
                      ["Optical mode order", pDepth],
                      ["Δφ @ 550 nm", `${((4 * Math.PI * n1 * thickness) / 550).toFixed(2)} rad`],
                      ["First min λ", `${Math.round((4 * n1 * thickness) / 3)} nm`],
                      ["First max λ", `${Math.round(2 * n1 * thickness)} nm`],
                    ].map(([label, val]) => (
                      <div key={label} className="flex justify-between mb-1.5">
                        <span className="opacity-40">{label}</span>
                        <span style={{ color: "#f5f5f5", opacity: 0.8 }}>{val}</span>
                      </div>
                    ))}
                  </div>
                </section>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
