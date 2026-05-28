import React, { useRef, useEffect } from 'react';
import { initCanvas, drawTitle } from './_canvas';

export interface StripEntry      { color: string; opacity?: number }
export interface StripLegendItem { color: string; label: string }
export interface CanvasStripProps {
  entries: StripEntry[];
  legend?: StripLegendItem[];
  title?:  string;
}

const PAD = { l: 8, r: 8, t: 18, b: 20 };

/** Convert 0-1 opacity to a two-character hex alpha suffix */
function opacityToHex(opacity: number): string {
  const clamped = Math.max(0, Math.min(1, opacity));
  const byte    = Math.round(clamped * 255);
  return byte.toString(16).padStart(2, '0');
}

export function CanvasStrip(props: CanvasStripProps) {
  const ref     = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef<() => void>(() => {});

  drawRef.current = () => {
    if (!ref.current) return;
    const r = initCanvas(ref.current);
    if (!r) return;
    const [ctx, W, H] = r;

    const { entries, legend = [], title } = props;

    const iW = W - PAD.l - PAD.r;
    const stripTop = PAD.t;
    const stripBot = H - PAD.b;
    const stripH   = stripBot - stripTop;

    if (entries.length > 0) {
      const entryW = iW / entries.length;
      for (let i = 0; i < entries.length; i++) {
        const e       = entries[i];
        const opacity = e.opacity ?? 1.0;
        const alpha   = opacityToHex(opacity);
        ctx.fillStyle = e.color + alpha;
        ctx.fillRect(PAD.l + i * entryW, stripTop, entryW, stripH);
      }
    }

    // Legend items at bottom, evenly spaced
    if (legend.length > 0) {
      const spacing = iW / legend.length;
      ctx.font = '8px monospace';
      for (let i = 0; i < legend.length; i++) {
        const item = legend[i];
        const cx   = PAD.l + i * spacing + spacing / 2;
        const ly   = H - 4;
        // Small colored rect
        const rectW = 8;
        const rectH = 6;
        ctx.fillStyle = item.color;
        ctx.fillRect(cx - spacing / 2 + 2, ly - rectH, rectW, rectH);
        // Label
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.textAlign = 'left';
        ctx.fillText(item.label, cx - spacing / 2 + rectW + 4, ly - 1);
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
