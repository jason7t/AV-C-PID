"""
data_loader.py  —  Load and preprocess DAM and BM datasets.

DAM split: test starts 2020-06-01 (confirmed from paper's knn_Q_DAM_1-12.csv row 0 = 24.65)
BM split:  last 17520 rows = 365 days × 48 half-hourly periods
"""
import warnings
import numpy as np
import pandas as pd
from config import (DAM_PATH, BM_PATH, DAM_TEST_START, DAM_TEST_END, BM_TEST_ROWS,
                    DAM_N_HORIZONS, BM_N_HORIZONS, CAL_FRAC)

warnings.filterwarnings('ignore')


def load_dam(path=DAM_PATH):
    """
    Load DAM data. Returns dict with X_all, y_all, n_pretrain, etc.
    DAM columns: EURPrices(+0)...EURPrices+23 (targets), then feature columns.
    Date format in CSV: dd/mm/yyyy HH:MM
    """
    df = pd.read_csv(path, index_col='DeliveryPeriod',
                     parse_dates=True,
                     date_parser=lambda d: pd.to_datetime(d, dayfirst=True))
    df = df.bfill().ffill()

    # Targets: first 24 columns
    target_cols = list(df.columns[:DAM_N_HORIZONS])
    feat_cols   = list(df.columns[DAM_N_HORIZONS:])

    for c in feat_cols + target_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.fillna(df.median(numeric_only=True))

    X = df[feat_cols].values.astype(np.float64)
    y = df[target_cols].values.astype(np.float64)

    # Date-based test split
    ts = pd.Timestamp(DAM_TEST_START)
    te = pd.Timestamp(DAM_TEST_END)
    # Handle timezone-naive index
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)

    test_mask = (idx >= ts) & (idx < te)
    pre_mask  = idx < ts

    X_pre,  y_pre  = X[pre_mask],  y[pre_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    n_pre = len(X_pre)
    X_all = np.vstack([X_pre, X_test])
    y_all = np.vstack([y_pre, y_test])

    # Cal split within pre-test (for SCP/QRA)
    n_cal = int(n_pre * CAL_FRAC)
    X_cal, y_cal   = X_pre[-n_cal:], y_pre[-n_cal:]
    X_train, y_train = X_pre[:-n_cal], y_pre[:-n_cal]

    print(f"[DAM] pre-test={n_pre}  test={len(X_test)} ({len(X_test)//24} days)  feat={X.shape[1]}")
    print(f"      Test: {df.index[pre_mask][-1].date()} → {df.index[test_mask][-1].date()}")

    return dict(
        df=df,
        X_all=X_all, y_all=y_all,
        X_train=X_train, y_train=y_train,
        X_cal=X_cal,   y_cal=y_cal,
        X_test=X_test, y_test=y_test,
        n_pretrain=n_pre,
        n_horizons=DAM_N_HORIZONS,
        market='DAM',
        target_col_names=[f'EURPrices+{i}' for i in range(DAM_N_HORIZONS)],
        feat_cols=feat_cols,
    )


def load_bm(path=BM_PATH):
    """
    Load BM data. Test = last 17520 rows (365 days × 48).
    BM columns: lag_2y...lag_17y (targets), then feature columns.
    Date format in CSV: mm/dd/yyyy HH:MM
    """
    df = pd.read_csv(path, na_values=['NULL'])
    df.columns = [c.lstrip('\ufeff') for c in df.columns]
    df['dt'] = pd.to_datetime(df['SettlementPeriod'], format='mixed', errors='coerce')
    df = df.sort_values('dt').reset_index(drop=True)

    # Drop carbon (x4) and gas (x5) columns, and index
    drop = [c for c in df.columns if 'x4' in c or 'x5' in c or c == 'index']
    df.drop(columns=drop, errors='ignore', inplace=True)

    target_cols = [f'lag_{i}y' for i in range(2, 2 + BM_N_HORIZONS)]
    feat_cols   = [c for c in df.columns if c not in target_cols
                   and c not in ['SettlementPeriod', 'dt']]

    for c in feat_cols + target_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[feat_cols + target_cols] = df[feat_cols + target_cols].fillna(
        df[feat_cols + target_cols].median())

    X = df[feat_cols].values.astype(np.float64)
    y = df[target_cols].values.astype(np.float64)

    # Position-based test split
    X_pre,  y_pre  = X[:-BM_TEST_ROWS], y[:-BM_TEST_ROWS]
    X_test, y_test = X[-BM_TEST_ROWS:], y[-BM_TEST_ROWS:]
    n_pre = len(X_pre)

    X_all = np.vstack([X_pre, X_test])
    y_all = np.vstack([y_pre, y_test])

    n_cal = int(n_pre * CAL_FRAC)
    X_cal, y_cal     = X_pre[-n_cal:], y_pre[-n_cal:]
    X_train, y_train = X_pre[:-n_cal], y_pre[:-n_cal]

    test_start = df['dt'].iloc[n_pre].date() if not pd.isna(df['dt'].iloc[n_pre]) else 'unknown'
    print(f"[BM]  pre-test={n_pre}  test={len(X_test)} ({len(X_test)//48} days)  feat={X.shape[1]}")
    print(f"      Test starts: {test_start}")

    return dict(
        df=df,
        X_all=X_all, y_all=y_all,
        X_train=X_train, y_train=y_train,
        X_cal=X_cal,   y_cal=y_cal,
        X_test=X_test, y_test=y_test,
        n_pretrain=n_pre,
        n_horizons=BM_N_HORIZONS,
        market='BM',
        target_col_names=target_cols,
        feat_cols=feat_cols,
    )
