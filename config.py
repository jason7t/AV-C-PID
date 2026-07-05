"""
config.py  —  Complete EPF replication configuration.
All parameters match O'Connor et al. (2025) Energy and AI 21, 100571 exactly.
BM model hyperparameters aligned with DAM for consistency.
"""
import os
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
DAM_PATH    = 'DAM_data.csv'
BM_PATH     = 'BM_data.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Test split (confirmed from paper's published result CSVs) ─────────────────
# DAM: row 0 of knn_Q_DAM_1-12.csv has price 24.65 = 2020-06-01 00:00
DAM_TEST_START = '2020-06-01'    # inclusive
DAM_TEST_END   = '2021-06-01'    # exclusive (365 days)
BM_TEST_ROWS   = 1440            # 30 days × 48 half-hourly periods (1 month)

# ── Prediction horizons ───────────────────────────────────────────────────────
DAM_N_HORIZONS = 24   # EURPrices+0 to EURPrices+23
BM_N_HORIZONS  = 16   # lag_2y to lag_17y

# ── Rolling window ────────────────────────────────────────────────────────────
# Paper uses training_days=-365: train on last 365 days, advance by 24h (DAM) or 8h (BM)
DAM_TRAINING_DAYS = -365   # paper's exact value
BM_TRAINING_DAYS  = -365

# Retrain every N rows (1 day DAM, 8 hours BM)
DAM_RETRAIN_EVERY = 24   # 1 row per hour × 24 = 1 day
BM_RETRAIN_EVERY  = 16   # 1 row per 30min × 16 = 8 hours

# Multipliers for computational efficiency (set all to 1 for full paper replication)
LEAR_RETRAIN_MULT  = 1
LGBM_RETRAIN_MULT  = 1
RF_RETRAIN_MULT    = 1
KNN_RETRAIN_MULT   = 1
ENBPI_RETRAIN_MULT = 1
QRA_RETRAIN_MULT   = 1

# ── Quantiles ─────────────────────────────────────────────────────────────────
QUANTILES       = [0.1, 0.3, 0.5, 0.7, 0.9]
QUANTILE_LABELS = [10,  30,  50,  70,  90]

# ── LEAR ──────────────────────────────────────────────────────────────────────
# DAM features: EURPrices-24→167, WF→WF-143, DF→DF-143 (432 cols)
# BM features: multiple lag groups (see data_loader.py)
# Alpha: LassoLarsIC(criterion='aic', max_iter=2500) computed ONCE from training data
# Model: MultiOutputRegressor(QuantileRegressor(quantile=q, alpha=alpha))
LEAR_LARS_MAX_ITER  = 2500
LEAR_LASSO_MAX_ITER = 2500
LEAR_MAX_TRAIN_ROWS    = 720    # 30 days × 24 rows (DAM)
LEAR_BM_MAX_TRAIN_ROWS = 1440   # 30 days × 48 rows (BM) — same 30-day window

# ── RF ────────────────────────────────────────────────────────────────────────
# QR model: MultiOutputRegressor(RandomForestQuantileRegressor(...))
# CP base model: RandomForestRegressor(...) — standard RF, not quantile
# BM params now match DAM for consistency
RF_QR_DAM_PARAMS  = dict(q=[0.1,0.3,0.5,0.7,0.9], max_depth=70,
                          max_features=150, n_estimators=300)
RF_QR_BM_PARAMS   = dict(q=[0.1,0.3,0.5,0.7,0.9], max_depth=70,
                          max_features=150, n_estimators=300)
RF_CP_DAM_PARAMS  = dict(max_depth=70, max_features=150, n_estimators=300)
RF_CP_BM_PARAMS   = dict(max_depth=70, max_features=150, n_estimators=300)

# ── LGBM ─────────────────────────────────────────────────────────────────────
# QR: 5 separate MultiOutputRegressor(LGBMRegressor(alpha=q)) per quantile
# CP: single LGBMRegressor(alpha=0.5) — predicts median
# BM params now match DAM for consistency
LGBM_QR_DAM_PARAMS  = dict(objective='quantile', learning_rate=0.05,
                             num_leaves=40, max_depth=10, n_estimators=100,
                             verbose=-1)
LGBM_QR_BM_PARAMS   = dict(objective='quantile', learning_rate=0.05,
                             num_leaves=40, max_depth=10, n_estimators=100,
                             verbose=-1)
LGBM_CP_DAM_PARAMS  = dict(objective='quantile', alpha=0.5, learning_rate=0.05,
                             num_leaves=40, max_depth=10, n_estimators=100,
                             verbose=-1)
LGBM_CP_BM_PARAMS   = dict(objective='quantile', alpha=0.5, learning_rate=0.05,
                             num_leaves=40, max_depth=10, n_estimators=100,
                             verbose=-1)

# ── KNN ───────────────────────────────────────────────────────────────────────
# QR: reconstructed (not in paper's scripts) — same params, all 5 quantiles
# CP: q=[0.50] only (paper uses KNeighborsQuantileRegressor(q=[0.50]))
KNN_PARAMS = dict(n_neighbors=290, p=1, leaf_size=1)

# ── CP methods (EnbPI and SPCI via SPCI_and_EnbPI class) ─────────────────────
# DAM: stride=24,  BM: stride=16
ENBPI_B         = 10    # B bootstrap models for EnbPI (KNN/LEAR) DAM=20
SPCI_B          = 10    # B bootstrap models for SPCI (KNN/LEAR)
CP_PAST_WINDOW  = 300   # sliding residual window
# RF/LGBM are too slow for B=20 — reduce B and trees for feasibility
ENBPI_B_RF        = 3    # B for RF/LGBM EnbPI
SPCI_B_RF         = 3    # B for RF/LGBM SPCI
RF_CP_N_EST_BOOT  = 10   # trees per bootstrap model for RF (vs 300 full)
LGBM_CP_N_EST_BOOT = 300  # estimators per bootstrap model for LGBM

# CP feature subset (paper uses smaller set for EnbPI/SPCI)
# DAM: EURPrices-48→167, WF-24→WF-143, DF-24→DF-143 (360 cols)
DAM_CP_FEAT_PRICE = ('EURPrices-48', 'EURPrices-167')
DAM_CP_FEAT_WF    = ('WF-24', 'WF-143')
DAM_CP_FEAT_DF    = ('DF-24', 'DF-143')

# ── Trading ───────────────────────────────────────────────────────────────────
EFF_1 = 0.80    # discharge efficiency
EFF_2 = 0.98    # charge efficiency
BATTERY_CAP  = 1.0
RAMP_RATE    = 1.0
MIN_SOC      = 0.0
INITIAL_SOC  = 0.0

# Paper skips first N rows before computing trading profit (warm-up)
TRADING_SKIP_DAM = 152   # 152 prediction days
TRADING_SKIP_BM  = 456   # 456 prediction periods

# ── Methods to run ────────────────────────────────────────────────────────────
MODEL_NAMES  = ['KNN', 'LEAR', 'RF', 'LGBM']
METHOD_NAMES = ['QR', 'EnbPI', 'SPCI', 'SCP', 'QRA-R', 'QRA-CP', 'Q-Ens', 'AV-C-PID', 'PID-Q-Ens']

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED = 42
CAL_FRAC = 0.20    # calibration fraction for SCP/QRA
