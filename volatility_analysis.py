# Volatility analysis tool for investigating forecast error patterns.
#
# Examines whether prediction errors are structurally larger during
# high-volatility periods, providing empirical motivation for the
# AV-C-PID method. Processes all result CSVs automatically or a
# single file when specified.
#
# Usage:
#   python volatility_analysis.py                        # all files
#   python volatility_analysis.py results/lear_QR_DAM.csv  # single file
#   python volatility_analysis.py --window 336           # custom window
#
# Output per file:
#   _timeline.png : prices, rolling volatility, and signed forecast error
#   _decile.png   : coverage and violation magnitude by volatility decile

import sys
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def load_predictions(path):
    import re
    df = pd.read_csv(path)

    actual_cols = [c for c in df.columns if '+' in c and 'Forecast' not in c]
    if not actual_cols:
        actual_cols = [c for c in df.columns
                       if re.fullmatch(r'lag_\d+y', c) and 'Forecast' not in c]
    if not actual_cols:
        return None  # not a predictions file — skip silently

    def get_q(lbl):
        cols = [f'{c}_Forecast_{lbl}' for c in actual_cols]
        if all(c in df.columns for c in cols):
            return df[cols].values.astype(float)
        return None

    actuals = df[actual_cols].values.astype(float)
    q10 = get_q(10);  q50 = get_q(50);  q90 = get_q(90)
    return actuals, q10, q50, q90, len(actual_cols), actual_cols


def flatten(arr):
    return arr.flatten() if arr is not None else None


def _corr(x, y, name):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 10:
        print(f"  {name}: not enough overlapping data"); return
    r, p_r = pearsonr(x[mask], y[mask])
    rho, p_rho = spearmanr(x[mask], y[mask])
    print(f"  {name:<30} Pearson r={r:+.3f} (p={p_r:.4f})  "
          f"Spearman ρ={rho:+.3f} (p={p_rho:.4f})")


def process_file(path, window, top_n):
    result = load_predictions(path)
    if result is None:
        return  # skip non-prediction files

    actuals, q10, q50, q90, n_h, actual_cols = result
    n_days = len(actuals)
    a_flat   = flatten(actuals)
    q10_flat = flatten(q10)
    q50_flat = flatten(q50)
    q90_flat = flatten(q90)
    n_t = len(a_flat)

    print(f"\n{'='*70}")
    print(f"File        : {path}")
    print(f"Days        : {n_days}  |  Horizons/day: {n_h}  |  Timeline: {n_t}")

    vol = pd.Series(a_flat).rolling(
        window=window, min_periods=max(4, window // 4),
        center=True).std().values

    if q50_flat is not None:
        signed_err = a_flat - q50_flat          # signed, shows direction
        abs_err    = np.abs(signed_err)         # for correlation / decile
    else:
        signed_err = abs_err = np.full(n_t, np.nan)
        print("  (no Q50 found — error series unavailable)")

    if q10_flat is not None and q90_flat is not None:
        below     = np.maximum(q10_flat - a_flat, 0.0)
        above     = np.maximum(a_flat  - q90_flat, 0.0)
        violation = below + above
        iw80      = q90_flat - q10_flat
        covered80 = ((a_flat >= q10_flat) & (a_flat <= q90_flat)).astype(float)
    else:
        violation = iw80 = covered80 = np.full(n_t, np.nan)
        print("  (no Q10/Q90 found — violation/coverage unavailable)")

    print("\n=== Correlation: rolling volatility vs prediction error ===")
    _corr(vol, abs_err,   "Volatility vs |error| (Q50)")
    _corr(vol, signed_err,"Volatility vs signed error (Q50)")
    _corr(vol, violation, "Volatility vs violation (80% PI)")
    _corr(vol, iw80,      "Volatility vs interval width (80%)")

    print("\n=== Mean error by volatility decile (1=calmest, 10=most volatile) ===")
    mask = ~np.isnan(vol) & ~np.isnan(abs_err)
    summary = None
    if mask.sum() > 20:
        deciles = pd.qcut(vol[mask], 10, labels=False, duplicates='drop')
        dec_df = pd.DataFrame({
            'decile':    deciles,
            'abs_err':   abs_err[mask],
            'signed_err':signed_err[mask],
            'cov80':     covered80[mask],
            'violation': violation[mask],
            'iw80':      iw80[mask],
            'vol':       vol[mask],
        })
        summary = dec_df.groupby('decile').mean()
        print(summary.to_string(float_format=lambda v: f"{v:7.3f}"))
        if not summary['cov80'].isna().all():
            cov_drop = summary['cov80'].iloc[0] - summary['cov80'].iloc[-1]
            print(f"\n  Target Cov80=0.800. "
                  f"Coverage drop calmest→most volatile: {cov_drop:+.3f}")

    print(f"\n=== Top {top_n} largest-error timestamps ===")
    if not np.all(np.isnan(abs_err)):
        top_idx = np.argsort(np.nan_to_num(abs_err, nan=-1))[::-1][:top_n]
        for idx in sorted(top_idx):
            day, hod = idx // n_h, idx % n_h
            print(f"  t={idx:5d} (day {day+1:3d}, step {hod:2d})  "
                  f"actual={a_flat[idx]:8.2f}  "
                  f"Q50={q50_flat[idx] if q50_flat is not None else float('nan'):8.2f}  "
                  f"signed_err={signed_err[idx]:+8.2f}  "
                  f"vol={vol[idx] if not np.isnan(vol[idx]) else float('nan'):7.2f}")

    t = np.arange(n_t)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(t, a_flat, color='steelblue', lw=0.8, label='Actual price')
    if q50_flat is not None:
        axes[0].plot(t, q50_flat, color='darkorange', lw=0.8, alpha=0.8, label='Q50 forecast')
    if q10_flat is not None and q90_flat is not None:
        axes[0].fill_between(t, q10_flat, q90_flat,
                             color='darkorange', alpha=0.15, label='Q10–Q90 interval')
    axes[0].set_ylabel('Price (€/MWh)')
    axes[0].set_title(f'{os.path.basename(path)} — full test-period timeline')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, vol, color='firebrick', lw=0.9)
    axes[1].set_ylabel(f'Rolling std\n(window={window})')
    axes[1].set_title('Market volatility (model-independent)')
    axes[1].grid(alpha=0.3)

    if not np.all(np.isnan(signed_err)):
        pos = np.where(signed_err >= 0, signed_err, 0)
        neg = np.where(signed_err <  0, signed_err, 0)
        axes[2].fill_between(t, 0, pos, color='darkgreen', alpha=0.6,
                             label='Under-prediction (actual > Q50)')
        axes[2].fill_between(t, 0, neg, color='firebrick', alpha=0.6,
                             label='Over-prediction (actual < Q50)')
        axes[2].axhline(0, color='black', lw=0.6, linestyle='--')
        axes[2].set_ylabel('Signed error (€/MWh)\nactual − Q50')
        axes[2].set_title('Signed forecast error — direction of bias')
        axes[2].legend(loc='upper right', fontsize=8)
    axes[2].set_xlabel('Timeline (sequential test steps)')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out1 = path.replace('.csv', '_timeline.png')
    plt.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out1}")

    if summary is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: coverage by decile
        if not summary['cov80'].isna().all():
            colors = ['firebrick' if c < 0.70 else ('darkorange' if c < 0.75 else 'steelblue')
                      for c in summary['cov80']]
            axes2[0].bar(summary.index + 1, summary['cov80'], color=colors)
            axes2[0].axhline(0.80, color='black', linestyle='--', lw=1, label='Target (80%)')
            axes2[0].set_xlabel('Volatility decile (1=calmest, 10=most volatile)')
            axes2[0].set_ylabel('Empirical Coverage80')
            axes2[0].set_title('Coverage vs volatility')
            axes2[0].legend(); axes2[0].grid(alpha=0.3, axis='y')

        # Panel B: violation magnitude by decile
        if not summary['violation'].isna().all():
            axes2[1].bar(summary.index + 1, summary['violation'], color='purple')
            axes2[1].set_xlabel('Volatility decile (1=calmest, 10=most volatile)')
            axes2[1].set_ylabel('Mean violation (80% PI, €/MWh)')
            axes2[1].set_title('Interval violation vs volatility')
            axes2[1].grid(alpha=0.3, axis='y')

        plt.suptitle(os.path.basename(path), fontsize=11, y=1.01)
        plt.tight_layout()
        out2 = path.replace('.csv', '_decile.png')
        plt.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {out2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file', nargs='?', default=None,
                    help='Path to a single results CSV (omit to process ALL '
                         'CSVs in results/ automatically)')
    ap.add_argument('--window', type=int, default=168,
                    help='Rolling volatility window in time steps '
                         '(default 168 = 7 days hourly; use 336 for 7-day BM)')
    ap.add_argument('--top-n', type=int, default=15,
                    help='Number of largest-error timestamps to print (default 15)')
    ap.add_argument('--results-dir', default='results',
                    help='Directory to scan when no file is given (default: results/)')
    args = ap.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.results_dir, '*.csv')))
        if not files:
            print(f"No CSV files found in '{args.results_dir}/'. "
                  f"Pass a specific file path or check --results-dir.")
            sys.exit(1)
        # Skip metric/trading summary files — only process prediction CSVs
        skip = {'DAM_metrics.csv','BM_metrics.csv','DAM_trading.csv','BM_trading.csv'}
        files = [f for f in files if os.path.basename(f) not in skip]
        # Skip already-generated timeline/decile PNGs' source CSVs if desired,
        # but actually process everything — user can delete unwanted outputs.
        print(f"Found {len(files)} prediction CSV(s) in '{args.results_dir}/'")

    for f in files:
        try:
            process_file(f, args.window, args.top_n)
        except Exception as e:
            print(f"\nERROR processing {f}: {e}")
            continue

    print("\n\nAll files processed.")


if __name__ == '__main__':
    main()
