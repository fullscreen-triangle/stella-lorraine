import React, { useRef, useEffect } from 'react';
import { initCanvas, drawTitle, drawGrid } from './_canvas';

export interface HistBand { lo: number; hi: number; color: string }
export interface CanvasHistogramProps {
  values:  number[];
  bands?:  HistBand[];
  bins?:   number;
  title?:  string;
  fmt?:    (v: number) => string;
  xMin?:   number;
  xMax?:   number;
}

const PAD = { l: 10, r: 10, t: 18, b: 26 };

export function CanvasHistogram(props: CanvasHistogramProps) {
  const ref     = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef<() => void>(() => {});

  drawRef.current = () => {
    if (!ref.current) return;
    const r = initCanvas(ref.current);
    if (!r) return;
    const [ctx, W, H] = r;

    const {
      values,
      bands = [],
      bins  = 40,
      title,
      fmt  = (v: number) => v.toFixed(2),
      xMin: xMinProp,
      xMax: xMaxProp,
    } = props;

    const iW = W - PAD.l - PAD.r;
    const iH = H - PAD.t - PAD.b;

    // Compute xMin / xMax
    let xMin: number;
    let xMax: number;

    if (bands.length > 0) {
      xMin = xMinProp ?? Math.min(...bands.map(b => b.lo));
      xMax = xMaxProp ?? Math.max(...bands.map(b => b.hi));
    } else {
      xMin = xMinProp ?? (values.length ? Math.min(...values) : 0);
      xMax = xMaxProp ?? (values.length ? Math.max(...values) : 1);
    }
    if (xMin === xMax) { xMin -= 1; xMax += 1; }

    const toX = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * iW;

    // Draw band background fills
    for (const band of bands) {
      const bx = toX(Math.max(band.lo, xMin));
      const bw = toX(Math.min(band.hi, xMax)) - bx;
      if (bw <= 0) continue;
      ctx.fillStyle = band.color + '20';
      ctx.fillRect(bx, PAD.t, bw, iH);
    }

    // Bin the values
    const binWidth = (xMax - xMin) / bins;
    const counts   = new Array<number>(bins).fill(0);
    for (const v of values) {
      if (v < xMin || v > xMax) continue;
      let idx = Math.floor((v - xMin) / binWidth);
      if (idx >= bins) idx = bins - 1;
      counts[idx]++;
    }
    const maxCount = Math.max(...counts, 1);

    // Helper: get color for bin midpoint
    const binColor = (binIdx: number): string => {
      const mid = xMin + (binIdx + 0.5) * binWidth;
      for (const band of bands) {
        if (mid >= band.lo && mid <= band.hi) return band.color + 'c0';
      }
      return '#4b5563c0';
    };

    // Draw bars
    const barW = iW / bins;
    for (let i = 0; i < bins; i++) {
      if (counts[i] === 0) continue;
      const barH = (counts[i] / maxCount) * iH;
      const bx   = PAD.l + i * barW;
      const by   = PAD.t + iH - barH;
      ctx.fillStyle = binColor(i);
      ctx.fillRect(bx, by, Math.max(barW - 0.5, 0.5), barH);
    }

    // Grid overlay
    drawGrid(ctx, PAD.l, PAD.t, iW, iH, 5, 4);

    // x-axis tick labels: xMin, midpoint, xMax
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '8px monospace';
    const baseline = H - 6;

    ctx.textAlign = 'left';
    ctx.fillText(fmt(xMin), PAD.l, baseline);

    ctx.textAlign = 'center';
    ctx.fillText(fmt((xMin + xMax) / 2), PAD.l + iW / 2, baseline);

    ctx.textAlign = 'right';
    ctx.fillText(fmt(xMax), PAD.l + iW, baseline);

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
