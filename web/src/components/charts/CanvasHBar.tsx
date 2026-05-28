import React, { useRef, useEffect } from 'react';
import { initCanvas, drawTitle, drawGrid } from './_canvas';

export interface HBarEntry { label: string; value: number; color: string }
export interface CanvasHBarProps {
  data:   HBarEntry[];
  title?: string;
}

const PAD = { l: 68, r: 14, t: 18, b: 12 };

export function CanvasHBar(props: CanvasHBarProps) {
  const ref     = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef<() => void>(() => {});

  drawRef.current = () => {
    if (!ref.current) return;
    const r = initCanvas(ref.current);
    if (!r) return;
    const [ctx, W, H] = r;

    const { data, title } = props;
    if (data.length === 0) return;

    const iW = W - PAD.l - PAD.r;
    const iH = H - PAD.t - PAD.b;

    const maxVal = Math.max(...data.map(d => d.value), 1);
    const rowH   = iH / data.length;

    // Grid (transposed: rows become cols for horizontal bars)
    drawGrid(ctx, PAD.l, PAD.t, iW, iH, 4, data.length);

    for (let i = 0; i < data.length; i++) {
      const entry  = data[i];
      const y      = PAD.t + i * rowH;
      const barLen = (entry.value / maxVal) * iW;
      const barY   = y + rowH * 0.1;
      const barH   = rowH * 0.8;

      // Background track
      ctx.fillStyle = entry.color + '20';
      ctx.fillRect(PAD.l, barY, iW, barH);

      // Filled bar
      ctx.fillStyle = entry.color;
      ctx.fillRect(PAD.l, barY, barLen, barH);

      // Label right-aligned before left edge
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '9px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(entry.label, PAD.l - 4, y + rowH / 2 + 3);

      // Value to the right of bar end
      if (entry.value > 0) {
        ctx.fillStyle = entry.color;
        ctx.font = '8px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(String(entry.value), PAD.l + barLen + 3, y + rowH / 2 + 3);
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
