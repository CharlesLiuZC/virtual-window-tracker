/** Frame-time governor, not a GPU profiler. Keep projection/CSS size unchanged. */
export class RenderBudget {
  scale = 1;
  private slowWindows = 0;
  private fastWindows = 0;

  reset() { this.scale = 1; this.slowWindows = this.fastWindows = 0; }

  observe(meanMs: number, p95Ms: number) {
    if (![meanMs, p95Ms].every(Number.isFinite) || meanMs <= 0) return this.scale;
    if (meanMs > 22 || p95Ms > 32) {
      this.slowWindows++;
      this.fastWindows = 0;
      if (this.slowWindows >= 2) { this.scale = Math.max(0.5, this.scale * 0.85); this.slowWindows = 0; }
    } else if (meanMs < 17 && p95Ms < 21) {
      this.fastWindows++;
      this.slowWindows = 0;
      if (this.fastWindows >= 6) { this.scale = Math.min(1, this.scale + 0.05); this.fastWindows = 0; }
    } else { this.slowWindows = this.fastWindows = 0; }
    return this.scale;
  }
}

export function renderPixelRatio(width: number, height: number, deviceRatio: number,
                                 cap: number, maxPixels: number, adaptiveScale: number) {
  const budgetRatio = Math.sqrt(Math.max(1, maxPixels) / Math.max(1, width * height));
  return Math.min(deviceRatio, cap, budgetRatio) * Math.min(1, Math.max(0.5, adaptiveScale));
}
