// shared canvas drawing utilities

export function initCanvas(
  canvas: HTMLCanvasElement
): [CanvasRenderingContext2D, number, number] | null {
  const W = (canvas.width  = canvas.offsetWidth);
  const H = (canvas.height = canvas.offsetHeight);
  if (W === 0 || H === 0) return null;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#070c09';
  ctx.fillRect(0, 0, W, H);
  return [ctx, W, H];
}

export function drawTitle(
  ctx: CanvasRenderingContext2D,
  title: string,
  x: number,
  y: number
) {
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(title, x, y);
}

export function drawGrid(
  ctx: CanvasRenderingContext2D,
  pl: number, pt: number, iW: number, iH: number,
  cols = 5, rows = 4
) {
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.5;
  for (let i = 1; i < cols; i++) {
    const x = pl + (iW * i) / cols;
    ctx.beginPath(); ctx.moveTo(x, pt); ctx.lineTo(x, pt + iH); ctx.stroke();
  }
  for (let i = 1; i < rows; i++) {
    const y = pt + (iH * i) / rows;
    ctx.beginPath(); ctx.moveTo(pl, y); ctx.lineTo(pl + iW, y); ctx.stroke();
  }
}
