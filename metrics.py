"""
metrics.py  —  Evaluation metrics for probabilistic forecasts.

APS   : Average Pinball Score (Σ pinball over 5 quantiles)
IW    : Interval Width (Q90-Q10)
Cov80 : Empirical 80% coverage (Q10-Q90)
Cov40 : Empirical 40% coverage (Q30-Q70)
Wink  : Winkler score
"""
import os
import numpy as np
import pandas as pd
from config import QUANTILES, QUANTILE_LABELS, RESULTS_DIR


def pinball(y, q_pred, q_level):
    e = y - q_pred
    return np.where(e >= 0, q_level * e, (q_level - 1) * e).mean()


def compute_all_metrics(y_test, preds, retrain_every=24):
    """
    y_test : (n, n_h)
    preds  : (n, n_h, 5)  → [Q10, Q30, Q50, Q70, Q90]

    Predictions repeat every retrain_every rows (one prediction per day).
    Subsample to every retrain_every-th row so predictions align with actuals.
    """
    # Subsample to aligned rows only (avoids repeated-prediction vs shifted-actual mismatch)
    y_test = y_test[::retrain_every]
    preds  = preds[::retrain_every]
    q10, q30, q50, q70, q90 = [preds[:, :, i] for i in range(5)]

    # APS: average pinball over 5 quantiles
    aps = np.mean([pinball(y_test, preds[:, :, i], q)
                   for i, q in enumerate(QUANTILES)])

    # Interval widths
    iw80 = (q90 - q10).mean()
    iw40 = (q70 - q30).mean()

    # Coverage
    cov80 = ((y_test >= q10) & (y_test <= q90)).mean()
    cov40 = ((y_test >= q30) & (y_test <= q70)).mean()

    # Winkler score (80% interval)
    alpha = 0.2
    below = (y_test < q10)
    above = (y_test > q90)
    inside = ~(below | above)
    winkler = ((q90 - q10) +
               (2 / alpha) * np.where(below, q10 - y_test, 0) +
               (2 / alpha) * np.where(above, y_test - q90, 0)).mean()

    return dict(APS=float(aps), IW80=float(iw80), IW40=float(iw40),
                Cov80=float(cov80), Cov40=float(cov40), Winkler=float(winkler))


def print_aps_table(all_results):
    """Print APS and Cov80 for all models and methods."""
    models  = list(all_results.keys())
    methods = list(next(iter(all_results.values())).keys()) if all_results else []
    if not models or not methods:
        return

    print(f"\n{'':8}", end='')
    for m in methods:
        print(f"  {m:>8}", end='')
    print()
    print('─' * (8 + len(methods) * 10))

    for model in models:
        print(f"  {model:<6}", end='')
        for method in methods:
            res = all_results.get(model, {}).get(method, {})
            aps = res.get('APS', float('nan'))
            print(f"  {aps:8.4f}", end='')
        print()

    print()
    print(f"{'':8}", end='')
    for m in methods:
        print(f"  {'Cov80':>8}", end='')
    print()
    for model in models:
        print(f"  {model:<6}", end='')
        for method in methods:
            res = all_results.get(model, {}).get(method, {})
            cov = res.get('Cov80', float('nan'))
            print(f"  {cov:8.4f}", end='')
        print()


def metrics_to_df(all_results):
    rows = []
    for model, mdict in all_results.items():
        for method, res in mdict.items():
            rows.append({'Model': model, 'Method': method, **res})
    return pd.DataFrame(rows).set_index(['Model', 'Method'])


def save_predictions_csv(preds, y_test, target_col_names, retrain_every,
                          model, method, market, results_dir=RESULTS_DIR):
    """
    Save prediction CSV in paper's format.
    One row per retrain step (i.e. one row per prediction day for DAM).
    Columns: actual values + Forecast_10/30/50/70/90 per horizon.
    """
    n_te, n_h, n_q = preds.shape
    n_h  = min(n_h, len(target_col_names))
    rows = {}

    # One row per batch start (paper produces 1 row per prediction day)
    batch_starts = list(range(0, n_te, retrain_every))

    for hi, col in enumerate(target_col_names[:n_h]):
        rows[col] = [float(y_test[bs, hi]) if bs < n_te else np.nan
                     for bs in batch_starts]

    for qi, lbl in enumerate(QUANTILE_LABELS):
        for hi, col in enumerate(target_col_names[:n_h]):
            rows[f'{col}_Forecast_{lbl}'] = [
                float(preds[bs, hi, qi]) if bs < n_te else np.nan
                for bs in batch_starts]

    df = pd.DataFrame(rows)
    fname = f"{model.lower()}_{method}_{market}.csv"
    fpath = os.path.join(results_dir, fname)
    df.to_csv(fpath, index=False)
    print(f"    Saved predictions → {fpath}  ({len(df)} rows × {len(df.columns)} cols)")
    return fpath
