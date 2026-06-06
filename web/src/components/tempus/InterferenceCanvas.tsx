import React, { useRef, useEffect, useCallback } from "react";
import type { ConstructScene } from "../../lib/tempus/construct";

// ── WebGL per-pixel interference engine ───────────────────────────────────────
// Renders I(p) = |Σ_k A_k · exp(i(k_k·|p − s_k| + φ_k))|² over a 2-D domain.
// The fragment shader computing this superposition IS the observation: the
// texture is the interference field. After drawing, the centre row is read back
// to measure fringe visibility and count — the GPU is the instrument.

const MAX = 32;

const VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main(){ v_uv = a_pos*0.5+0.5; gl_Position = vec4(a_pos,0.0,1.0); }`;

const FRAG = `
precision highp float;
varying vec2 v_uv;
uniform int   u_n;
uniform vec2  u_domain;        // world size (metres)
uniform vec2  u_pos[${MAX}];   // source positions (metres)
uniform float u_amp[${MAX}];
uniform float u_phase[${MAX}];
uniform float u_k[${MAX}];      // 2π/λ
uniform float u_invSumAmp2;     // 1 / (Σ|A|)²

vec3 ramp(float t){
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.02,0.05,0.05);
  vec3 c1 = vec3(0.05,0.22,0.28);
  vec3 c2 = vec3(0.35,0.90,0.85);   // teal #58E6D9
  vec3 c3 = vec3(0.96,0.62,0.20);   // amber
  vec3 c4 = vec3(0.99,0.98,0.95);
  if(t<0.25) return mix(c0,c1,t*4.0);
  if(t<0.50) return mix(c1,c2,(t-0.25)*4.0);
  if(t<0.75) return mix(c2,c3,(t-0.50)*4.0);
  return mix(c3,c4,(t-0.75)*4.0);
}

void main(){
  vec2 p = (v_uv - 0.5) * u_domain;   // world coordinates
  float re = 0.0, im = 0.0;
  for(int i=0;i<${MAX};i++){
    if(i>=u_n) break;
    float d = distance(p, u_pos[i]);
    float theta = u_k[i]*d + u_phase[i];
    re += u_amp[i]*cos(theta);
    im += u_amp[i]*sin(theta);
  }
  float I = (re*re + im*im) * u_invSumAmp2;   // normalised intensity [0,1]
  gl_FragColor = vec4(ramp(pow(I, 0.85)), 1.0);
}`;

export interface Observables {
  visibility: number;
  fringes: number;
  sources: number;
  intensity_max: number;
  intensity_min: number;
}

export function InterferenceCanvas({ scene, onObserve }: { scene: ConstructScene | null; onObserve?: (o: Observables) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const progRef = useRef<WebGLProgram | null>(null);
  const obsRef = useRef(onObserve);
  obsRef.current = onObserve;

  // one-time GL init
  const ensureGL = useCallback((): boolean => {
    if (glRef.current) return true;
    const canvas = ref.current;
    if (!canvas) return false;
    const gl = canvas.getContext("webgl", { antialias: true, preserveDrawingBuffer: true });
    if (!gl) return false;
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src); gl.compileShader(s);
      return s;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return false;
    gl.useProgram(prog);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    glRef.current = gl;
    progRef.current = prog;
    return true;
  }, []);

  const draw = useCallback(() => {
    const canvas = ref.current;
    if (!canvas || !scene) return;
    if (!ensureGL()) return;
    const gl = glRef.current!, prog = progRef.current!;

    const W = (canvas.width = canvas.offsetWidth || 320);
    const H = (canvas.height = canvas.offsetHeight || 240);
    if (W === 0 || H === 0) return;
    gl.viewport(0, 0, W, H);
    gl.useProgram(prog);

    const waves = scene.waves.slice(0, MAX);
    const n = waves.length;
    const pos = new Float32Array(MAX * 2);
    const amp = new Float32Array(MAX);
    const phase = new Float32Array(MAX);
    const kArr = new Float32Array(MAX);
    let sumAmp = 0;
    waves.forEach((w, i) => {
      pos[i * 2] = w.x; pos[i * 2 + 1] = w.y;
      amp[i] = w.amp; phase[i] = w.phase; kArr[i] = (2 * Math.PI) / w.wavelength;
      sumAmp += Math.abs(w.amp);
    });
    const u = (name: string) => gl.getUniformLocation(prog, name);
    gl.uniform1i(u("u_n"), n);
    gl.uniform2f(u("u_domain"), scene.domain.w, scene.domain.h);
    gl.uniform2fv(u("u_pos"), pos);
    gl.uniform1fv(u("u_amp"), amp);
    gl.uniform1fv(u("u_phase"), phase);
    gl.uniform1fv(u("u_k"), kArr);
    gl.uniform1f(u("u_invSumAmp2"), sumAmp > 0 ? 1 / (sumAmp * sumAmp) : 1);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // ── readback: centre row → visibility + fringe count ──────────────────────
    if (obsRef.current) {
      const midY = Math.floor(H / 2);
      const px = new Uint8Array(W * 4);
      gl.readPixels(0, midY, W, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      let lo = 1, hi = 0;
      const lum: number[] = new Array(W);
      for (let i = 0; i < W; i++) {
        const l = (0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2]) / 255;
        lum[i] = l; if (l < lo) lo = l; if (l > hi) hi = l;
      }
      // count fringe maxima above mid-threshold with a small separation
      const thr = lo + 0.5 * (hi - lo);
      let fringes = 0, lastPeak = -10;
      for (let i = 2; i < W - 2; i++) {
        if (lum[i] > thr && lum[i] >= lum[i - 1] && lum[i] > lum[i + 1] && i - lastPeak > 3) {
          fringes++; lastPeak = i;
        }
      }
      const visibility = hi + lo > 1e-6 ? (hi - lo) / (hi + lo) : 0;
      obsRef.current({ visibility, fringes, sources: n, intensity_max: hi, intensity_min: lo });
    }
  }, [scene, ensureGL]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(c);
    return () => ro.disconnect();
  }, [draw]);

  return <canvas ref={ref} style={{ width: "100%", height: "100%", display: "block", background: "#02100e" }} />;
}
