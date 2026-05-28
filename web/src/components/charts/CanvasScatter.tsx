import React, { useRef, useEffect } from 'react';
import { initCanvas, drawTitle, drawGrid } from './_canvas';

export interface ScatterBand  { label: string; lo: number; hi: number; color: string }
export interface ScatterPoint { y: number; color: string; r?: number }
export interface CanvasScatterProps {
  points: ScatterPoint[];
  bands?:  ScatterBand[];
  title?:  string;
  fmt?:    (v: number) => string;
  yMin?:   number;
  yMax?:   number;
}

const PAD = { l: 54, r: 10, t: 16, b: 26 };

export function CanvasScatter(props: CanvasScatterProps) {
  const ref     = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef<() => void>(() => {});

  drawRef.current = () => {
    if (!ref.current) return;
    const r = initCanvas(ref.current);
    if (!r) return;
    const [ctx, W, H] = r;

    const { points, bands = [], title, fmt = (v: number) => v.toFixed(2), yMin: yMinProp, yMax: yMaxProp } = props;

    const iW = W - PAD.l - PAD.r;
    const iH = H - PAD.t - PAD.b;

    // Compute yMin / yMax
    let yMin = yMinProp;
    let yMax = yMaxProp;

    if (yMin === undefined || yMax === undefined) {
      const allVals: number[] = [
        ...points.map(p => p.y),
        ...bands.flatMap(b => [b.lo, b.hi]),
      ];
      if (yMin === undefined) yMin = allVals.length ? Math.min(...allVals) : 0;
      if (yMax === undefined) yMax = allVals.length ? Math.max(...allVals) : 1;
    }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }

    const toY = (v: number) => PAD.t + iH - ((v - yMin!) / (yMax! - yMin!)) * iH;

    // Draw bands
    for (const band of bands) {
      const yTop = toY(band.hi);
      const yBot = toY(band.lo);
      ctx.fillStyle = band.color + '20';
      ctx.fillRect(PAD.l, yTop, iW, yBot - yTop);
      // top stroke line
      ctx.strokeStyle = band.color + '80';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(PAD.l, yTop);
      ctx.lineTo(PAD.l + iW, yTop);
      ctx.stroke();
      // band label at left edge
      ctx.fillStyle = band.color;
      ctx.font = '8px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(band.label, PAD.l + 2, yTop + 9);
    }

    // Grid
    drawGrid(ctx, PAD.l, PAD.t, iW, iH, 5, 4);

    // Points
    const n = points.length;
    for (let i = 0; i < n; i++) {
      const pt = points[i];
      const px = n > 1 ? PAD.l + (i / (n - 1)) * iW : PAD.l + iW / 2;
      const py = toY(pt.y);
      const radius = pt.r ?? 2.5;
      ctx.fillStyle = pt.color;
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    // y-axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '8px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(fmt(yMax!), PAD.l - 4, PAD.t + 5);
    ctx.fillText(fmt(yMin!), PAD.l - 4, PAD.t + iH);

    // x-axis labels
    ctx.textAlign = 'left';
    ctx.fillText('0', PAD.l, H - 6);
    ctx.textAlign = 'right';
    ctx.fillText(String(n), PAD.l + iW, H - 6);

    // Title
    if (title) drawTitle(ctx, title, PAD.l + 4, PAD.t - 3);
  };

  useEffect(() => { drawRef.current(); });

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ro = new ResizeObserver(() => drawRef.current());
    ro.observe(c);
    return () => ro.disconnect();
  }, []);

  return <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block' }} />;
}
