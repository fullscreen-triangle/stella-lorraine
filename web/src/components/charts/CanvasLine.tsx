import React, { useRef, useEffect } from 'react';
import { initCanvas, drawTitle, drawGrid } from './_canvas';

export interface LineSeries { values: number[]; color: string; fill?: boolean; label?: string }
export interface CanvasLineProps {
  series:  LineSeries[];
  title?:  string;
  yMin?:   number;
  yMax?:   number;
  fmt?:    (v: number) => string;
}

const PAD = { l: 36, r: 10, t: 18, b: 24 };

export function CanvasLine(props: CanvasLineProps) {
  const ref     = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef<() => void>(() => {});

  drawRef.current = () => {
    if (!ref.current) return;
    const r = initCanvas(ref.current);
    if (!r) return;
    const [ctx, W, H] = r;

    const {
      series,
      title,
      yMin: yMinProp,
      yMax: yMaxProp,
      fmt = (v: number) => v.toFixed(2),
    } = props;

    if (series.length === 0) return;

    const iW = W - PAD.l - PAD.r;
    const iH = H - PAD.t - PAD.b;

    // Compute yMin / yMax across all series
    let yMin = yMinProp;
    let yMax = yMaxProp;

    if (yMin === undefined || yMax === undefined) {
      const allVals = series.flatMap(s => s.values);
      if (yMin === undefined) yMin = allVals.length ? Math.min(...allVals) : 0;
      if (yMax === undefined) yMax = allVals.length ? Math.max(...allVals) : 1;
    }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }

    const toY = (v: number) => PAD.t + iH - ((v - yMin!) / (yMax! - yMin!)) * iH;
    const toX = (i: number, len: number) =>
      len > 1 ? PAD.l + (i / (len - 1)) * iW : PAD.l + iW / 2;

    // Grid
    drawGrid(ctx, PAD.l, PAD.t, iW, iH, 5, 4);

    // Draw each series
    for (const s of series) {
      if (s.values.length === 0) continue;
      const len = s.values.length;

      // Build path
      const buildPath = () => {
        ctx.beginPath();
        for (let i = 0; i < len; i++) {
          const x = toX(i, len);
          const y = toY(s.values[i]);
          if (i === 0) ctx.moveTo(x, y);
          else         ctx.lineTo(x, y);
        }
      };

      // Fill polygon
      if (s.fill) {
        ctx.beginPath();
        for (let i = 0; i < len; i++) {
          const x = toX(i, len);
          const y = toY(s.values[i]);
          if (i === 0) ctx.moveTo(x, y);
          else         ctx.lineTo(x, y);
        }
        // Close down to baseline
        ctx.lineTo(toX(len - 1, len), PAD.t + iH);
        ctx.lineTo(toX(0, len),        PAD.t + iH);
        ctx.closePath();
        ctx.fillStyle = s.color + '20';
        ctx.fill();
      }

      // Stroke line
      buildPath();
      ctx.strokeStyle = s.color;
      ctx.lineWidth   = 1.5;
      ctx.lineJoin    = 'round';
      ctx.stroke();
    }

    // y-axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '8px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(fmt(yMax!), PAD.l - 4, PAD.t + 5);
    ctx.fillText('0', PAD.l - 4, PAD.t + iH);

    // Title
    if (title) drawTitle(ctx, title, PAD.l + 4, PAD.t - 5);
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
