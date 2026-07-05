# Quantile regression base models for all four regressors.
#
# Implements LEAR, RF, LGBM, and KNN following O'Connor et al. (2025).
# Each model is fitted separately per quantile level to produce the
# five quantile forecasts used throughout the pipeline.
#
# Models:
#   LEAR : MinMaxScaler + LassoLarsIC alpha + QuantileRegressor
#   RF   : MultiOutputRegressor(RandomForestQuantileRegressor)
#   LGBM : 5 x MultiOutputRegressor(LGBMRegressor(alpha=q))
#   KNN  : KNeighborsRegressor, n_neighbors=290, p=1, leaf_size=1
#
# O'Connor, C., Collins, J., Prestwich, S., and Visentin, A. (2025).
# Conformal Prediction for Electricity Price Forecasting in Day-Ahead
# and Balancing Markets. Energy and AI, 21, 100571.
# https://doi.org/10.1016/j.egyai.2025.100571

import time, warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import QuantileRegressor, LassoLarsIC, Lasso
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')

from config import (QUANTILES, QUANTILE_LABELS,
                    DAM_RETRAIN_EVERY, BM_RETRAIN_EVERY,
                    LEAR_RETRAIN_MULT, RF_RETRAIN_MULT,
                    LGBM_RETRAIN_MULT, KNN_RETRAIN_MULT,
                    LEAR_LARS_MAX_ITER, LEAR_LASSO_MAX_ITER,
                    LEAR_MAX_TRAIN_ROWS, LEAR_BM_MAX_TRAIN_ROWS,
                    RF_QR_DAM_PARAMS, RF_QR_BM_PARAMS,
                    RF_CP_DAM_PARAMS, RF_CP_BM_PARAMS,
                    LGBM_QR_DAM_PARAMS, LGBM_QR_BM_PARAMS,
                    LGBM_CP_DAM_PARAMS, LGBM_CP_BM_PARAMS,
                    KNN_PARAMS,
                    RF_CP_N_EST_BOOT, LGBM_CP_N_EST_BOOT, SEED)



def select_lear_features_dam(df_or_array, feat_cols=None):
    if isinstance(df_or_array, pd.DataFrame):
        df = df_or_array
        prices = df.loc[:, 'EURPrices-24':'EURPrices-167'].values
        wind   = df.loc[:, 'WF':'WF-143'].values
        demand = df.loc[:, 'DF':'DF-143'].values
        return prices, wind, demand
    else:
        X = df_or_array
        prices = X[:, 0:144]
        wind   = X[:, 144:288]    # WF to WF-143
        demand = X[:, 312:456]    # DF to DF-143
        return prices, wind, demand


def select_cp_features_dam(df_or_array):
    if isinstance(df_or_array, pd.DataFrame):
        df = df_or_array
        p = df.loc[:, 'EURPrices-48':'EURPrices-167'].values
        w = df.loc[:, 'WF-24':'WF-143'].values
        d = df.loc[:, 'DF-24':'DF-143'].values
        return np.hstack([p, w, d])
    else:
        X = df_or_array
        p = X[:, 24:144]      # EURPrices-48 to EURPrices-167
        w = X[:, 168:288]     # WF-24 to WF-143
        d = X[:, 336:456]     # DF-24 to DF-143
        return np.hstack([p, w, d])



def compute_lear_alpha(X_all_train, y_all_train, market='DAM'):

    # Use last 30 days = same window as LEAR QR
    max_rows = LEAR_MAX_TRAIN_ROWS if market == 'DAM' else LEAR_BM_MAX_TRAIN_ROWS
    X_win = X_all_train[-max_rows:]
    y_win = y_all_train[-max_rows:]

    if market == 'DAM':
        prices, wind, demand = select_lear_features_dam(X_win)
        sc1 = preprocessing.MinMaxScaler().fit(prices)
        sc2 = preprocessing.MinMaxScaler().fit(wind)
        sc3 = preprocessing.MinMaxScaler().fit(demand)
        X_scaled = np.hstack([sc1.transform(prices),
                               sc2.transform(wind),
                               sc3.transform(demand)])
    else:
        X_scaled = X_win
    y_sc = preprocessing.MinMaxScaler().fit(y_win)
    y_scaled = y_sc.transform(y_win)
    try:
        alpha = LassoLarsIC(criterion='aic', max_iter=LEAR_LARS_MAX_ITER
                            ).fit(X_scaled, y_scaled[:, 0]).alpha_
        if alpha <= 0 or np.isnan(alpha):
            alpha = 0.01
    except Exception:
        alpha = 0.01
    return float(alpha)


def _lear_prepare_dam(X_tr, y_tr, X_te):
    prices_tr, wind_tr, demand_tr = select_lear_features_dam(X_tr)
    prices_te, wind_te, demand_te = select_lear_features_dam(X_te)

    sc1 = preprocessing.MinMaxScaler().fit(prices_tr)
    sc2 = preprocessing.MinMaxScaler().fit(wind_tr)
    sc3 = preprocessing.MinMaxScaler().fit(demand_tr)
    y_sc = preprocessing.MinMaxScaler().fit(y_tr)

    X_tr_s = np.hstack([sc1.transform(prices_tr),
                         sc2.transform(wind_tr),
                         sc3.transform(demand_tr)])
    X_te_s = np.hstack([sc1.transform(prices_te),
                         sc2.transform(wind_te),
                         sc3.transform(demand_te)])
    y_tr_s = y_sc.transform(y_tr)
    return X_tr_s, y_tr_s, X_te_s, y_sc


def fit_predict_lear(X_tr, y_tr, X_te, alpha, market='DAM'):
    if market == 'DAM':
        X_tr_s, y_tr_s, X_te_s, y_sc = _lear_prepare_dam(X_tr, y_tr, X_te)
    else:
        sc_x = preprocessing.MinMaxScaler().fit(X_tr)
        sc_y = preprocessing.MinMaxScaler().fit(y_tr)
        X_tr_s = sc_x.transform(X_tr)
        X_te_s = sc_x.transform(X_te)
        y_tr_s = sc_y.transform(y_tr)
        y_sc   = sc_y

    n_te, n_h = len(X_te_s), y_tr.shape[1]
    preds_s = np.zeros((n_te, n_h, len(QUANTILES)))

    for qi, q in enumerate(QUANTILES):
        model = MultiOutputRegressor(
            QuantileRegressor(quantile=q, alpha=alpha, solver='highs-ds'),
            n_jobs=-1)
        model.fit(X_tr_s, y_tr_s)
        preds_s[:, :, qi] = model.predict(X_te_s)

    # Inverse-transform
    preds = np.zeros_like(preds_s)
    for qi in range(len(QUANTILES)):
        preds[:, :, qi] = y_sc.inverse_transform(preds_s[:, :, qi])

    return preds.astype(np.float32)



def fit_predict_rf(X_tr, y_tr, X_te, market='DAM'):
    from quantile_forest import RandomForestQuantileRegressor as QF_RF
    params = RF_QR_DAM_PARAMS.copy() if market == 'DAM' else RF_QR_BM_PARAMS.copy()
    q_levels = params.pop('q')

    model = QF_RF(n_jobs=-1, random_state=SEED, **params)
    model.fit(X_tr, y_tr)
    raw = model.predict(X_te, quantiles=q_levels)  # (n_te, n_h, n_q)
    return raw.astype(np.float32)


def fit_predict_lgbm(X_tr, y_tr, X_te, market='DAM'):
    params = LGBM_QR_DAM_PARAMS.copy() if market == 'DAM' else LGBM_QR_BM_PARAMS.copy()
    n_te, n_h = len(X_te), y_tr.shape[1]
    preds = np.zeros((n_te, n_h, len(QUANTILES)), dtype=np.float32)

    for qi, q in enumerate(QUANTILES):
        p = {**params, 'alpha': q}
        # n_jobs=1 — LightGBM uses its own threading internally;
        # combining with joblib multiprocessing causes SIGSEGV on macOS.
        model = MultiOutputRegressor(lgb.LGBMRegressor(**p), n_jobs=1)
        model.fit(X_tr, y_tr)
        preds[:, :, qi] = model.predict(X_te).astype(np.float32)

    return preds


def fit_predict_knn(X_tr, y_tr, X_te, market='DAM'):
    from sklearn.neighbors import NearestNeighbors
    k = min(KNN_PARAMS['n_neighbors'], len(X_tr))
    nn = NearestNeighbors(n_neighbors=k, p=KNN_PARAMS['p'],
                          leaf_size=KNN_PARAMS['leaf_size'],
                          algorithm='ball_tree', n_jobs=-1)
    nn.fit(X_tr)
    _, idx = nn.kneighbors(X_te)
    neighbor_y = y_tr[idx]  
    preds = np.quantile(neighbor_y, QUANTILES, axis=1)  
    return preds.transpose(1, 2, 0).astype(np.float32)  


def get_cp_base_model(model_name, market='DAM'):
    from sklearn.ensemble import RandomForestRegressor
    if model_name == 'KNN':
        # Use standard KNeighborsRegressor instead of KNeighborsQuantileRegressor:
        # q=[0.50] prediction = mean of K neighbours ≈ KNeighborsRegressor default.
        # Avoids sklearn_quantile._validate_data compatibility issue with newer sklearn.
        from sklearn.neighbors import KNeighborsRegressor
        k = min(KNN_PARAMS['n_neighbors'], 99999)  # cap handled at fit time
        return KNeighborsRegressor(n_neighbors=k, p=KNN_PARAMS['p'],
                                   leaf_size=KNN_PARAMS['leaf_size'],
                                   algorithm='ball_tree', n_jobs=-1)
    elif model_name == 'LEAR':
        # alpha will be set at fit time
        return None   # special handling in cp_methods.py
    elif model_name == 'RF':
        p = RF_CP_DAM_PARAMS.copy() if market == 'DAM' else RF_CP_BM_PARAMS.copy()
        # Use RF_CP_N_EST_BOOT trees: 300 trees × B=20 × 24 strides is infeasible.
        # 10 trees per bootstrap model keeps the same algorithm at feasible speed.
        p['n_estimators'] = RF_CP_N_EST_BOOT
        return RandomForestRegressor(n_jobs=-1, random_state=SEED, **p)
    elif model_name == 'LGBM':
        p = LGBM_CP_DAM_PARAMS.copy() if market == 'DAM' else LGBM_CP_BM_PARAMS.copy()
        p['n_estimators'] = LGBM_CP_N_EST_BOOT
        return lgb.LGBMRegressor(**p)
    else:
        raise ValueError(f"Unknown model: {model_name}")


_FIT_PREDICT = {
    'KNN':  fit_predict_knn,
    'LEAR': fit_predict_lear,
    'RF':   fit_predict_rf,
    'LGBM': fit_predict_lgbm,
}

_MULT = {
    'KNN':  'KNN_RETRAIN_MULT',
    'LEAR': 'LEAR_RETRAIN_MULT',
    'RF':   'RF_RETRAIN_MULT',
    'LGBM': 'LGBM_RETRAIN_MULT',
}


def run_qr(model_name, data, quantiles=None, retrain_every=None):
    from config import (LEAR_RETRAIN_MULT, RF_RETRAIN_MULT,
                        LGBM_RETRAIN_MULT, KNN_RETRAIN_MULT)
    _mults = {'KNN': KNN_RETRAIN_MULT, 'LEAR': LEAR_RETRAIN_MULT,
              'RF': RF_RETRAIN_MULT,  'LGBM': LGBM_RETRAIN_MULT}

    if quantiles is None:
        quantiles = QUANTILES
    market = data['market']
    base   = DAM_RETRAIN_EVERY if market == 'DAM' else BM_RETRAIN_EVERY
    if retrain_every is None:
        retrain_every = base * _mults.get(model_name, 1)

    X_all, y_all = data['X_all'], data['y_all']
    n_pre  = data['n_pretrain']
    n_te   = len(data['X_test'])
    n_h    = data['n_horizons']
    fp     = _FIT_PREDICT[model_name]

    # Max training window — LEAR uses shorter window (30 days) to keep
    # LP solver fast: 8760 rows × 432 features = ~35s/LP vs ~2.5s for 720 rows.
    # 120 LP solves × 35s = 70 min/step without the cap.
    if model_name == 'LEAR':
        max_rows = LEAR_MAX_TRAIN_ROWS if market == 'DAM' else LEAR_BM_MAX_TRAIN_ROWS
    else:
        max_rows = 365 * (24 if market == 'DAM' else 48)

    # Pre-compute LEAR alpha once from initial training data
    lear_alpha = None
    if model_name == 'LEAR':
        X_init = X_all[:n_pre]
        y_init = y_all[:n_pre]
        print(f"  Computing LEAR alpha from {len(X_init)} training rows...")
        lear_alpha = compute_lear_alpha(X_init, y_init, market)
        print(f"  LEAR alpha = {lear_alpha:.6f}")

    preds  = np.zeros((n_te, n_h, len(quantiles)), dtype=np.float32)
    steps  = list(range(0, n_te, retrain_every))
    n_step = len(steps)
    t0_all = time.time()

    print(f"  [QR-{model_name} | {market}] {n_step} retrains (every {retrain_every} rows)")

    for si, bs in enumerate(steps):
        be      = min(bs + retrain_every, n_te)
        win_end = n_pre + bs
        win_st  = max(0, win_end - max_rows)
        X_tr    = X_all[win_st:win_end]
        y_tr    = y_all[win_st:win_end]
        X_te    = X_all[n_pre + bs : n_pre + be]

        pct = 100.0 * (si + 1) / n_step
        t0  = time.time()
        print(f"    step {si+1:3d}/{n_step}  ({pct:5.1f}%)  "
              f"train[{win_st}:{win_end}] ({win_end-win_st} rows) ...",
              end='', flush=True)

        if model_name == 'LEAR':
            batch = fp(X_tr, y_tr, X_te, lear_alpha, market)
        else:
            batch = fp(X_tr, y_tr, X_te, market)

        preds[bs:be] = batch[:be-bs]
        print(f"  {time.time()-t0:.1f}s")

    print(f"  [QR-{model_name}] done. Total: {(time.time()-t0_all)/60:.1f} min\n")
    return preds
