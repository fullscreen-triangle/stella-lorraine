import React, { useRef, useEffect } from 'react';
import { initCanvas, drawTitle, drawGrid } from './_canvas';

export interface BarEntry { label: string; value: number; color: string }
export interface CanvasBarProps {
  data:        BarEntry[];
  title?:      string;
  showValues?: boolean;
}

const PAD = { l: 8, r: 8, t: 18, b: 34 };

export function CanvasBar(props: CanvasBarProps) {
  const ref     = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef<() => void>(() => {});

  drawRef.current = () => {
    if (!ref.current) return;
    const r = initCanvas(ref.current);
    if (!r) return;
    const [ctx, W, H] = r;

    const { data, title, showValues = true } = props;
    if (data.length === 0) return;

    const iW = W - PAD.l - PAD.r;
    const iH = H - PAD.t - PAD.b;

    const maxVal = Math.max(...data.map(d => d.value), 1);
    const colW   = iW / data.length;

    // Grid
    drawGrid(ctx, PAD.l, PAD.t, iW, iH, data.length, 4);

    for (let i = 0; i < data.length; i++) {
      const entry  = data[i];
      const x      = PAD.l + i * colW;
      const barH   = (entry.value / maxVal) * iH;
      const barX   = x + colW * 0.1;
      const barW   = colW * 0.8;
      const barY   = PAD.t + iH - barH;

      // Background track
      ctx.fillStyle = entry.color + '20';
      ctx.fillRect(barX, PAD.t, barW, iH);

      // Filled bar
      ctx.fillStyle = entry.color;
      ctx.fillRect(barX, barY, barW, barH);

      // Label below bar (truncated to 8 chars)
      const labelText = entry.label.length > 8 ? entry.label.slice(0, 8) : entry.label;
      ctx.fillStyle = entry.color;
      ctx.font = '8px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(labelText, x + colW / 2, H - PAD.b + 12);

      // Value count above bar
      if (showValues && entry.value > 0) {
        ctx.fillStyle = 'rgba(255,255,255,0.65)';
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(String(entry.value), x + colW / 2, barY - 3);
      }
    }

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
