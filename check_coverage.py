"""
check_coverage.py  —  Quick standalone evaluation for one prediction file.

Place in the same folder as results/ and run: python check_coverage.py
Change FILE_NAME at the top to check any prediction file.

Metrics reported:
  Coverage 80%  : fraction of actuals inside Q10–Q90  (target 0.80)
  Coverage 40%  : fraction of actuals inside Q30–Q70  (target 0.40)
  APS           : Average Pinball Score over 5 quantiles (lower = better)
  IW80          : Mean interval width Q90–Q10 (narrower = sharper)
  IW40          : Mean interval width Q70–Q30
  Winkler80     : Winkler score for 80% interval (lower = better)
  MAE Q50       : Mean absolute error of Q50 point forecast
  RMSE Q50      : Root mean squared error of Q50
"""
import os, math
import numpy as np
import pandas as pd

# ── Manual selection ──────────────────────────────────────────────────────────
FILE_NAME = 'lgbm_Q_1-12'   # ← change this to any model_method_market
# ─────────────────────────────────────────────────────────────────────────────

path = os.path.join('results', FILE_NAME + '.csv')
df   = pd.read_csv(path)

# Detect target columns
actual_cols = [c for c in df.columns if '+' in c and 'Forecast' not in c]
if not actual_cols:
    # BM target columns follow the exact pattern lag_{N}y (e.g. lag_2y,
    # lag_17y) — NOT lag_{N}x{M} feature columns. The naive 'lag_' prefix
    # match also caught hundreds of feature columns (lag_-3x1, lag_2x7,
    # etc.), causing get_q() to silently return None for every quantile.
    import re
    actual_cols = [c for c in df.columns
                   if re.fullmatch(r'lag_\d+y', c) and 'Forecast' not in c]
n_h, n_r = len(actual_cols), len(df)

actuals = df[actual_cols].values.astype(float)  # (n_rows, n_h)

def get_q(lbl):
    cols = [f'{c}_Forecast_{lbl}' for c in actual_cols]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None
    return df[cols].values.astype(float)

q10 = get_q(10);  q30 = get_q(30);  q50 = get_q(50)
q70 = get_q(70);  q90 = get_q(90)

# ── Coverage ──────────────────────────────────────────────────────────────────
inside_80 = (actuals >= q10) & (actuals <= q90)
inside_40 = (actuals >= q30) & (actuals <= q70)
cov_80    = inside_80.mean()
cov_40    = inside_40.mean()

# ── Interval widths ───────────────────────────────────────────────────────────
iw80 = (q90 - q10).mean()
iw40 = (q70 - q30).mean()

# ── APS (average pinball score) ───────────────────────────────────────────────
def pinball(y, q_hat, tau):
    e = y - q_hat
    return np.where(e >= 0, tau * e, (tau - 1) * e).mean()

quantiles = [(q10, 0.1), (q30, 0.3), (q50, 0.5), (q70, 0.7), (q90, 0.9)]
aps = np.mean([pinball(actuals, q, tau) for q, tau in quantiles if q is not None])

# ── Winkler score (80% interval) ─────────────────────────────────────────────
alpha  = 0.2
below  = actuals < q10
above  = actuals > q90
winkler = ((q90 - q10)
           + (2/alpha) * np.where(below, q10 - actuals, 0)
           + (2/alpha) * np.where(above, actuals - q90, 0)).mean()

# ── Point forecast quality (Q50) ─────────────────────────────────────────────
if q50 is not None:
    mae  = np.abs(actuals - q50).mean()
    rmse = np.sqrt(((actuals - q50)**2).mean())
else:
    mae = rmse = float('nan')

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\nFile    : {path}")
print(f"Rows    : {n_r}  |  Horizons: {n_h}")
print()
print(f"{'Metric':<20} {'Value':>10}  {'Target':>8}  {'Diff':>10}")
print("─" * 56)
print(f"{'Coverage 80%':<20} {cov_80:>10.4f}  {'0.8000':>8}  {cov_80-0.80:>+10.4f}")
print(f"{'Coverage 40%':<20} {cov_40:>10.4f}  {'0.4000':>8}  {cov_40-0.40:>+10.4f}")
print(f"{'APS':<20} {aps:>10.4f}  {'lower=better':>8}")
print(f"{'IW80 (Q10-Q90)':<20} {iw80:>10.4f}  {'narrower=better':>8}")
print(f"{'IW40 (Q30-Q70)':<20} {iw40:>10.4f}")
print(f"{'Winkler 80%':<20} {winkler:>10.4f}  {'lower=better':>8}")
print(f"{'MAE (Q50)':<20} {mae:>10.4f}")
print(f"{'RMSE (Q50)':<20} {rmse:>10.4f}")

# ── Per-horizon coverage ──────────────────────────────────────────────────────
print()
print("Per-horizon coverage 80% (Q10–Q90):")
for h, c in enumerate(inside_80.mean(axis=0)):
    bar  = '█' * int(c * 20)
    flag = ' ← low' if c < 0.70 else (' ← high' if c > 0.90 else '')
    print(f"  h={h:>2}  {c:.3f}  {bar}{flag}")

print()
print("Per-horizon coverage 40% (Q30–Q70):")
for h, c in enumerate(inside_40.mean(axis=0)):
    bar  = '█' * int(c * 20)
    flag = ' ← low' if c < 0.30 else (' ← high' if c > 0.50 else '')
    print(f"  h={h:>2}  {c:.3f}  {bar}{flag}")
