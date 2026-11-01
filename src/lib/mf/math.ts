export type NavRow = { date: string; nav: string };

export type MonthNav = { ym: string; nav: number };

export type Point = { ym: string; fund: number; bench: number };

export function monthEndNavs(rows: NavRow[]): MonthNav[] {
  const byMonth = new Map<string, { time: number; nav: number }>();
  for (const row of rows) {
    const parts = row.date.split("-").map(Number);
    if (parts.length !== 3 || parts.some((n) => Number.isNaN(n))) continue;
    const [day, month, year] = parts;
    const nav = Number(row.nav);
    if (!Number.isFinite(nav) || nav <= 0) continue;
    const ym = `${year}-${String(month).padStart(2, "0")}`;
    const time = Date.UTC(year, month - 1, day);
    const prev = byMonth.get(ym);
    if (!prev || time > prev.time) byMonth.set(ym, { time, nav });
  }
  return [...byMonth.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([ym, value]) => ({ ym, nav: value.nav }));
}

export function monthlyReturns(navs: MonthNav[]): { ym: string; ret: number }[] {
  const out: { ym: string; ret: number }[] = [];
  for (let i = 1; i < navs.length; i++) {
    const prev = navs[i - 1].nav;
    if (prev <= 0) continue;
    out.push({ ym: navs[i].ym, ret: navs[i].nav / prev - 1 });
  }
  return out;
}

export function alignReturns(
  fund: { ym: string; ret: number }[],
  bench: { ym: string; ret: number }[],
  months: number,
): Point[] {
  const benchBy = new Map(bench.map((row) => [row.ym, row.ret]));
  const paired: Point[] = [];
  for (const row of fund) {
    const b = benchBy.get(row.ym);
    if (b === undefined) continue;
    paired.push({ ym: row.ym, fund: row.ret, bench: b });
  }
  if (paired.length <= months) return paired;
  return paired.slice(paired.length - months);
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sampleStdev(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  const sq = values.reduce((sum, value) => sum + (value - m) ** 2, 0);
  return Math.sqrt(sq / (values.length - 1));
}

export function ols(y: number[], x: number[]): { alpha: number; beta: number; r2: number } {
  const n = Math.min(y.length, x.length);
  if (n < 3) return { alpha: 0, beta: 0, r2: 0 };
  const mx = mean(x.slice(0, n));
  const my = mean(y.slice(0, n));
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (x[i] - mx) ** 2;
    sxy += (x[i] - mx) * (y[i] - my);
  }
  const beta = sxx === 0 ? 0 : sxy / sxx;
  const alpha = my - beta * mx;
  let sse = 0;
  let sst = 0;
  for (let i = 0; i < n; i++) {
    sse += (y[i] - (alpha + beta * x[i])) ** 2;
    sst += (y[i] - my) ** 2;
  }
  const r2 = sst === 0 ? 0 : 1 - sse / sst;
  return { alpha, beta, r2 };
}

export function compounded(returns: number[]): number {
  return returns.reduce((product, value) => product * (1 + value), 1);
}

export function annualisedFromMonthly(total: number, months: number): number {
  if (months <= 0 || total <= 0) return 0;
  return total ** (12 / months) - 1;
}

export type IndexStats = {
  beta: number;
  r2: number;
  te: number;
  active: number;
  ir: number;
  upCapture: number | null;
  downCapture: number | null;
  upMonths: number;
  downMonths: number;
  upBeta: number | null;
  downBeta: number | null;
};

export function indexStats(points: Point[]): IndexStats | null {
  if (points.length < 6) return null;
  const y = points.map((p) => p.fund);
  const x = points.map((p) => p.bench);
  const fit = ols(y, x);
  const activeMonthly = points.map((p) => p.fund - p.bench);
  const te = sampleStdev(activeMonthly) * Math.sqrt(12);
  const active = annualisedFromMonthly(compounded(y) / compounded(x), points.length);
  const ir = te === 0 ? 0 : active / te;
  const up = points.filter((p) => p.bench > 0);
  const down = points.filter((p) => p.bench < 0);
  const capture = (rows: Point[]) => {
    if (rows.length < 2) return null;
    const b = compounded(rows.map((p) => p.bench)) - 1;
    if (Math.abs(b) < 1e-8) return null;
    return (compounded(rows.map((p) => p.fund)) - 1) / b;
  };
  const sideBeta = (rows: Point[]) => {
    if (rows.length < 4) return null;
    return ols(
      rows.map((p) => p.fund),
      rows.map((p) => p.bench),
    ).beta;
  };
  return {
    beta: fit.beta,
    r2: fit.r2,
    te,
    active,
    ir,
    upCapture: capture(up),
    downCapture: capture(down),
    upMonths: up.length,
    downMonths: down.length,
    upBeta: sideBeta(up),
    downBeta: sideBeta(down),
  };
}

export type Drawdown = {
  maxDd: number;
  peakYm: string;
  troughYm: string;
  recoveryMonths: number | null;
  recovered: boolean;
};

export function drawdown(points: Point[]): Drawdown | null {
  if (points.length < 2) return null;
  let wealth = 1;
  let peak = 1;
  let peakYm = points[0].ym;
  let maxDd = 0;
  let worstPeakYm = peakYm;
  let troughYm = points[0].ym;
  let troughIndex = 0;
  let worstPeakIndex = 0;
  let peakIndex = 0;
  const path: number[] = [];
  points.forEach((point, index) => {
    wealth *= 1 + point.fund;
    path.push(wealth);
    if (wealth >= peak) {
      peak = wealth;
      peakYm = point.ym;
      peakIndex = index;
    }
    const dd = wealth / peak - 1;
    if (dd < maxDd) {
      maxDd = dd;
      worstPeakYm = peakYm;
      worstPeakIndex = peakIndex;
      troughYm = point.ym;
      troughIndex = index;
    }
  });
  const peakAtStart = path[worstPeakIndex] ?? peak;
  let recovered = false;
  let recoveryMonths: number | null = null;
  for (let i = troughIndex + 1; i < path.length; i++) {
    if (path[i] >= peakAtStart) {
      recovered = true;
      recoveryMonths = i - troughIndex;
      break;
    }
  }
  return { maxDd, peakYm: worstPeakYm, troughYm, recoveryMonths, recovered };
}

export function annualisedVol(returns: number[]): number {
  return sampleStdev(returns) * Math.sqrt(12);
}

export function growthOf(amount: number, returns: number[]): number {
  return amount * compounded(returns);
}

export function pct(value: number | null, digits = 1): string {
  if (value === null || Number.isNaN(value)) return "\u2014";
  return `${(value * 100).toFixed(digits)}%`;
}

export function inr(value: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}
