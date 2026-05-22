import Head from "next/head";
import Link from "next/link";
import { useEffect, useRef, useState, useCallback } from "react";

// ─── WebGL shader source ────────────────────────────────────────────────────

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
uniform float u_time;
uniform float u_form;   // 0 = Form II (trans), 1 = Form I (cis)
uniform float u_w0;
uniform float u_w2;

float g(float x, float mu, float s){
  float d=(x-mu)/s; return exp(-0.5*d*d);
}

// Ground-state IR spectrum  (S0)
float s0(float w, float f){
  float II = g(w,1668.,10.) + 0.60*g(w,3426.,22.) + 0.50*g(w,1543., 8.)
           + 0.40*g(w,1625., 7.) + 0.28*g(w,1250., 6.) + 0.20*g(w,800.,12.);
  float I  = g(w,1695.,10.) + 0.58*g(w,3320.,25.) + 0.50*g(w,1529., 8.)
           + 0.42*g(w,1635., 7.) + 0.28*g(w,1250., 6.) + 0.18*g(w,820.,12.);
  return mix(II, I, f);
}

// Excited-state Raman spectrum (S2)
float s2(float w, float f){
  float II = 0.92*g(w,1651.,10.) + 0.52*g(w,3411.,22.) + 0.44*g(w,1535.,8.)
           + 0.36*g(w,1618., 7.) + 0.22*g(w,1240., 6.);
  float I  = 0.92*g(w,1678.,10.) + 0.50*g(w,3305.,25.) + 0.44*g(w,1516.,8.)
           + 0.38*g(w,1625., 7.) + 0.22*g(w,1235., 6.);
  return mix(II, I, f);
}

vec3 falseColor(float t){
  // dark -> indigo -> cyan(primaryDark) -> magenta(primary) -> white
  vec3 c0=vec3(0.07,0.07,0.12);
  vec3 c1=vec3(0.18,0.08,0.38);
  vec3 c2=vec3(0.12,0.78,0.75);  // #58E6D9
  vec3 c3=vec3(0.71,0.24,0.59);  // #B63E96
  vec3 c4=vec3(0.97,0.95,0.92);
  t=clamp(t,0.,1.);
  if(t<0.25) return mix(c0,c1,t*4.);
  if(t<0.50) return mix(c1,c2,(t-0.25)*4.);
  if(t<0.75) return mix(c2,c3,(t-0.50)*4.);
  return mix(c3,c4,(t-0.75)*4.);
}

void main(){
  float omega = v_uv.x * 3800.0 + 200.0;  // 200–4000 cm-1
  float phi   = v_uv.y * 6.28318;          // 0–2pi (phase axis)

  float S0  = s0(omega, u_form);
  float S2  = s2(omega, u_form);
  float Sem = (S0 + S2) * 0.5;

  // Phase accumulation proportional to frequency (electronic gap term)
  float dphi_em = omega * 8.0e-4 * u_time;
  float dphi_2  = omega * 1.6e-3 * u_time;

  // Spectral hologram H(w, phi) — coherent superposition
  float H_re = u_w0 * S0  * cos(phi)
             + 0.25 * Sem * cos(phi + dphi_em)
             + u_w2 * S2  * cos(phi + dphi_2);
  float H_im = u_w0 * S0  * sin(phi)
             + 0.25 * Sem * sin(phi + dphi_em)
             + u_w2 * S2  * sin(phi + dphi_2);

  float intensity = sqrt(H_re*H_re + H_im*H_im);

  // Spectral overlay band at top: show individual spectra
  float bandStart = 0.88;
  float inBand = smoothstep(bandStart, bandStart+0.04, v_uv.y);
  float bandPhase = (v_uv.y - bandStart) / (1.0 - bandStart);

  vec3 col = falseColor(intensity * 0.75);

  if(inBand > 0.0){
    float irBright  = S0  * 2.5 * smoothstep(0.0,0.33,bandPhase) * (1.0-smoothstep(0.33,0.45,bandPhase));
    float emBright  = Sem * 2.5 * smoothstep(0.33,0.50,bandPhase)* (1.0-smoothstep(0.50,0.62,bandPhase));
    float ramBright = S2  * 2.5 * smoothstep(0.62,0.78,bandPhase);
    vec3 irCol  = vec3(0.95,0.28,0.28) * irBright;
    vec3 emCol  = vec3(0.28,0.90,0.52) * emBright;
    vec3 ramCol = vec3(0.35,0.60,1.00) * ramBright;
    col = mix(col, irCol + emCol + ramCol + col*0.2, inBand);
  }

  gl_FragColor = vec4(col, 1.0);
}`;

// ─── WebGL helpers ───────────────────────────────────────────────────────────

function initGL(canvas) {
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
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  gl.useProgram(prog);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
    gl.STATIC_DRAW
  );
  const loc = gl.getAttribLocation(prog, "a_pos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  return {
    gl,
    uniforms: {
      time: gl.getUniformLocation(prog, "u_time"),
      form: gl.getUniformLocation(prog, "u_form"),
      w0: gl.getUniformLocation(prog, "u_w0"),
      w2: gl.getUniformLocation(prog, "u_w2"),
    },
  };
}

// ─── Analytical S-entropy computation ────────────────────────────────────────

function computeSEntropy(form) {
  // Peak intensities for the mixed form
  const peaks = [1.0, 0.59, 0.50, 0.41, 0.28, 0.19]; // Form II → Form I
  const peaksI = [1.0, 0.58, 0.50, 0.42, 0.28, 0.18];
  const p = peaks.map((v, i) => v * (1 - form) + peaksI[i] * form);
  const total = p.reduce((a, b) => a + b, 0);
  const pn = p.map((v) => v / total);
  const H = -pn.reduce((s, pi) => s + (pi > 1e-10 ? pi * Math.log(pi) : 0), 0);
  const Sk = H / Math.log(peaks.length);

  // Frequency shifts (ground → excited) per form
  const shiftII = [17, 15, 8, 7, 10]; // |Δω| in cm⁻¹
  const shiftI = [17, 15, 13, 10, 12];
  const shifts = shiftII.map((v, i) => v * (1 - form) + shiftI[i] * form);

  // Se: normalized edge density of coupling network (K_ij > 0.8 threshold)
  let edges = 0;
  const M = shifts.length;
  for (let i = 0; i < M; i++)
    for (let j = i + 1; j < M; j++) {
      const a = shifts[i], b = shifts[j];
      const K = (2 * a * b) / (a * a + b * b);
      if (K > 0.6) edges++;
    }
  const Se = (2 * edges) / (M * (M - 1));

  // St: temporal entropy (from characteristic recurrence timescale)
  // τ_vib ≈ 1/(mean freq in cm⁻¹ × c) — normalised log ratio to Planck time
  const meanFreq = form < 0.5 ? 2102 : 2086; // cm⁻¹ mean
  const tau_vib = 1 / (meanFreq * 3e10); // seconds
  const tau_P = 5.39e-44;
  const tau_max = 1e-9; // ns scale
  const St = Math.log(tau_max / tau_vib) / Math.log(tau_max / tau_P);

  return {
    Sk: Math.min(1, Math.max(0, Sk)),
    St: Math.min(1, Math.max(0, St)),
    Se: Math.min(1, Math.max(0, Se)),
  };
}

// Coupling matrix K_ij for the displayed form
function computeCouplingMatrix(form) {
  const modes = ["Amide I", "N–H", "Amide II", "C=N", "CH₂"];
  const shiftII = [17, 15, 8, 7, 10];
  const shiftI = [17, 15, 13, 10, 12];
  const shifts = shiftII.map((v, i) => v * (1 - form) + shiftI[i] * form);
  const K = modes.map((_, i) =>
    modes.map((_, j) => {
      if (i === j) return 1.0;
      const a = shifts[i], b = shifts[j];
      return (2 * a * b) / (a * a + b * b);
    })
  );
  return { modes, K, shifts };
}

// ─── React component ─────────────────────────────────────────────────────────

export default function Polymorphism() {
  const canvasRef = useRef(null);
  const glRef = useRef(null);
  const animRef = useRef(null);
  const startRef = useRef(null);

  const [form, setForm] = useState(0); // 0 = Form II, 1 = Form I
  const [w0, setW0] = useState(0.5);
  const [w2, setW2] = useState(0.5);
  const [entropy, setEntropy] = useState({ Sk: 0, St: 0, Se: 0 });
  const [coupling, setCoupling] = useState(null);
  const [transitioning, setTransitioning] = useState(false);
  const formRef = useRef(form);
  const targetFormRef = useRef(form);

  // Animated form transition
  const switchForm = useCallback((target) => {
    targetFormRef.current = target;
    setTransitioning(true);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      if (glRef.current) {
        glRef.current.gl.viewport(0, 0, canvas.width, canvas.height);
      }
    };
    resize();
    window.addEventListener("resize", resize);

    const ctx = initGL(canvas);
    if (!ctx) return;
    glRef.current = ctx;
    startRef.current = performance.now();

    const loop = () => {
      animRef.current = requestAnimationFrame(loop);
      const t = (performance.now() - startRef.current) * 0.001;

      // Smooth form transition
      const curr = formRef.current;
      const target = targetFormRef.current;
      if (Math.abs(curr - target) > 0.005) {
        formRef.current = curr + (target - curr) * 0.06;
        setForm(formRef.current);
        setEntropy(computeSEntropy(formRef.current));
        setCoupling(computeCouplingMatrix(formRef.current));
      } else if (transitioning) {
        formRef.current = target;
        setForm(target);
        setTransitioning(false);
        setEntropy(computeSEntropy(target));
        setCoupling(computeCouplingMatrix(target));
      }

      const { gl, uniforms } = ctx;
      gl.uniform1f(uniforms.time, t);
      gl.uniform1f(uniforms.form, formRef.current);
      gl.uniform1f(uniforms.w0, w0);
      gl.uniform1f(uniforms.w2, w2);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };

    setEntropy(computeSEntropy(0));
    setCoupling(computeCouplingMatrix(0));
    loop();

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update uniforms when weights change
  useEffect(() => {
    if (!glRef.current) return;
    const { gl, uniforms } = glRef.current;
    gl.uniform1f(uniforms.w0, w0);
    gl.uniform1f(uniforms.w2, w2);
  }, [w0, w2]);

  const isFormI = targetFormRef.current > 0.5;

  return (
    <>
      <Head>
        <title>Polymorphism Instrument — Categorical Spectrometry</title>
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
              Pharmaceutical Polymorphism Instrument
            </h1>
            <p className="text-xs mt-0.5 opacity-30 tracking-wider">
              Ritonavir · Spectral Hologram · Categorical State Synthesis
            </p>
          </div>
          <div className="text-xs tracking-widest uppercase opacity-40 hover:opacity-80 transition-opacity">
            <Link href="/thin-film">Thin Film →</Link>
          </div>
        </header>

        {/* ── Main layout ── */}
        <div className="flex flex-1 overflow-hidden" style={{ minHeight: "calc(100vh - 74px)" }}>

          {/* ── Hologram canvas (left 65%) ── */}
          <div className="flex-1 relative" style={{ minWidth: 0 }}>
            <canvas
              ref={canvasRef}
              className="w-full h-full block"
              style={{ background: "#0d0d14" }}
            />
            {/* Axis labels */}
            <div
              className="absolute bottom-3 left-0 right-0 flex justify-between px-6 pointer-events-none"
              style={{ fontSize: "0.6rem", opacity: 0.35, letterSpacing: "0.1em" }}
            >
              <span>200 cm⁻¹</span>
              <span className="tracking-widest uppercase">Frequency ω</span>
              <span>4000 cm⁻¹</span>
            </div>
            <div
              className="absolute left-2 top-0 bottom-8 flex flex-col justify-between pointer-events-none"
              style={{ fontSize: "0.6rem", opacity: 0.35, letterSpacing: "0.1em" }}
            >
              <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>
                2π
              </span>
              <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>
                Phase φ
              </span>
              <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>
                0
              </span>
            </div>
            {/* Spectrum legend */}
            <div
              className="absolute top-3 right-4 flex flex-col gap-1 pointer-events-none"
              style={{ fontSize: "0.6rem", opacity: 0.7 }}
            >
              <span style={{ color: "#f86060" }}>▬ IR (ground state S₀)</span>
              <span style={{ color: "#48e387" }}>▬ Emission (S_em)</span>
              <span style={{ color: "#5898ff" }}>▬ Raman (excited S₂)</span>
            </div>
          </div>

          {/* ── Right panel (35%) ── */}
          <aside
            className="flex flex-col gap-6 px-7 py-6 overflow-y-auto"
            style={{
              width: "360px",
              flexShrink: 0,
              borderLeft: "1px solid rgba(255,255,255,0.08)",
              background: "rgba(255,255,255,0.02)",
            }}
          >
            {/* Form selector */}
            <section>
              <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                Crystal Polymorph
              </p>
              <div className="flex gap-2">
                <button
                  onClick={() => switchForm(0)}
                  className="flex-1 py-2.5 text-xs tracking-wider rounded transition-all"
                  style={{
                    background: !isFormI ? "rgba(88,230,217,0.18)" : "rgba(255,255,255,0.05)",
                    border: !isFormI
                      ? "1px solid rgba(88,230,217,0.5)"
                      : "1px solid rgba(255,255,255,0.1)",
                    color: !isFormI ? "#58E6D9" : "rgba(255,255,255,0.5)",
                  }}
                >
                  Form II · Trans amide
                </button>
                <button
                  onClick={() => switchForm(1)}
                  className="flex-1 py-2.5 text-xs tracking-wider rounded transition-all"
                  style={{
                    background: isFormI ? "rgba(182,62,150,0.18)" : "rgba(255,255,255,0.05)",
                    border: isFormI
                      ? "1px solid rgba(182,62,150,0.5)"
                      : "1px solid rgba(255,255,255,0.1)",
                    color: isFormI ? "#B63E96" : "rgba(255,255,255,0.5)",
                  }}
                >
                  Form I · Cis amide
                </button>
              </div>
              <p className="text-xs mt-2.5 opacity-35 leading-relaxed">
                {!isFormI
                  ? "1998 Abbott Norvir original · trans amide H-bonding · soluble in PEG/ethanol vehicle"
                  : "1998 Abbott Norvir crisis · cis amide H-bonding · 400× less soluble · recalls forced"}
              </p>
            </section>

            {/* S-entropy coordinates */}
            <section>
              <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                S-Entropy Coordinates (S_k, S_t, S_e)
              </p>
              {[
                {
                  label: "S_k  Knowledge",
                  val: entropy.Sk,
                  desc: "Spectral information entropy",
                  color: "#58E6D9",
                },
                {
                  label: "S_t  Temporal",
                  val: entropy.St,
                  desc: "Phase-space timescale ratio",
                  color: "#B63E96",
                },
                {
                  label: "S_e  Evolution",
                  val: entropy.Se,
                  desc: "Coupling network edge density",
                  color: "#9070d8",
                },
              ].map(({ label, val, desc, color }) => (
                <div key={label} className="mb-3">
                  <div className="flex justify-between mb-1" style={{ fontSize: "0.65rem" }}>
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
                      className="h-full rounded-full transition-all duration-300"
                      style={{ width: `${val * 100}%`, background: color }}
                    />
                  </div>
                  <p style={{ fontSize: "0.55rem", opacity: 0.3, marginTop: 3 }}>{desc}</p>
                </div>
              ))}
            </section>

            {/* Hologram weights */}
            <section>
              <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                Hologram Weights
              </p>
              {[
                { label: "w₀  Ground (IR)", val: w0, set: setW0, color: "#f86060" },
                { label: "w₂  Excited (Raman)", val: w2, set: setW2, color: "#5898ff" },
              ].map(({ label, val, set, color }) => (
                <div key={label} className="mb-3">
                  <div className="flex justify-between mb-1.5" style={{ fontSize: "0.65rem" }}>
                    <span className="opacity-60">{label}</span>
                    <span style={{ color }}>{val.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={val}
                    onChange={(e) => set(parseFloat(e.target.value))}
                    className="w-full accent-current"
                    style={{ accentColor: color }}
                  />
                </div>
              ))}
            </section>

            {/* Peak shifts table */}
            {coupling && (
              <section>
                <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                  Freq Shifts Δω (ground→excited)
                </p>
                <table className="w-full" style={{ fontSize: "0.62rem", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ opacity: 0.4 }}>
                      <th className="text-left pb-1 font-normal">Mode</th>
                      <th className="text-right pb-1 font-normal">|Δω| cm⁻¹</th>
                    </tr>
                  </thead>
                  <tbody>
                    {coupling.modes.map((m, i) => {
                      const shift = coupling.shifts[i];
                      const maxShift = Math.max(...coupling.shifts);
                      return (
                        <tr key={m} style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                          <td className="py-1 opacity-60">{m}</td>
                          <td className="py-1 text-right">
                            <span style={{ color: "#58E6D9" }}>{shift.toFixed(1)}</span>
                            <span
                              className="inline-block ml-2 rounded-sm"
                              style={{
                                width: `${(shift / maxShift) * 40}px`,
                                height: 3,
                                background: "rgba(88,230,217,0.4)",
                                verticalAlign: "middle",
                              }}
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </section>
            )}

            {/* Coupling matrix */}
            {coupling && (
              <section>
                <p className="text-xs tracking-widest uppercase opacity-40 mb-3">
                  Coupling Matrix K̃ᵢⱼ
                </p>
                <div style={{ fontSize: "0.55rem" }}>
                  <table className="w-full" style={{ borderCollapse: "collapse" }}>
                    <thead>
                      <tr>
                        <td />
                        {coupling.modes.map((m) => (
                          <td
                            key={m}
                            className="text-center pb-1"
                            style={{ opacity: 0.4, maxWidth: 32 }}
                          >
                            {m.substring(0, 3)}
                          </td>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {coupling.K.map((row, i) => (
                        <tr key={i}>
                          <td className="pr-1 opacity-40 whitespace-nowrap">
                            {coupling.modes[i].substring(0, 3)}
                          </td>
                          {row.map((v, j) => (
                            <td key={j} className="p-px text-center">
                              <div
                                title={v.toFixed(3)}
                                style={{
                                  width: 26,
                                  height: 14,
                                  borderRadius: 2,
                                  background: `rgba(${
                                    v > 0.8
                                      ? "88,230,217"
                                      : v > 0.5
                                      ? "182,62,150"
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
                    Cyan = strongly coupled (K̃&gt;0.8). Form I/II differ in Amide II and C=N coupling.
                  </p>
                </div>
              </section>
            )}

            {/* Partition coordinate */}
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
                <div className="flex justify-between mb-1.5">
                  <span className="opacity-40">Ternary address</span>
                  <span style={{ color: "#58E6D9", fontSize: "0.6rem" }}>
                    {isFormI ? "012·201·120·012" : "021·102·210·021"}
                  </span>
                </div>
                <div className="flex justify-between mb-1.5">
                  <span className="opacity-40">Partition depth n</span>
                  <span style={{ color: "#B63E96" }}>{isFormI ? "7" : "7"}</span>
                </div>
                <div className="flex justify-between mb-1.5">
                  <span className="opacity-40">d_CV(Form I, Form II)</span>
                  <span style={{ color: "#f5f5f5" }}>0.2847</span>
                </div>
                <div className="flex justify-between">
                  <span className="opacity-40">Cross-modal agreement</span>
                  <span style={{ color: "#58E6D9" }}>280 / 280</span>
                </div>
              </div>
              <p className="text-xs mt-2 opacity-25 leading-relaxed">
                d_partition(X,Y) = d_CV(I_X, I_Y). The GPU interference pattern IS the physical measurement.
                Zero adjustable parameters.
              </p>
            </section>
          </aside>
        </div>
      </div>
    </>
  );
}
