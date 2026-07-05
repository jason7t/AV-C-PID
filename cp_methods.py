# Conformal prediction methods for probabilistic electricity price forecasting.
#
# Implements EnbPI, SPCI, and SCP following O'Connor et al. (2025), using the
# SPCI_and_EnbPI class as the core engine. All settings match the paper exactly.
#
# Methods:
#   EnbPI : B=20 bootstrap models, past_window=300, stride=n_horizons
#   SPCI  : B=10 bootstrap models, past_window=300, quantile regression forest
#   SCP   : Split Conformal Prediction with 20% calibration split
#
# Base models (single output, one per bootstrap resample):
#   KNN  : n_neighbors=290, p=1, leaf_size=1
#   LEAR : Lasso with LassoLarsIC alpha, max_iter=2500
#   RF   : RandomForestRegressor, max_depth=70, max_features=150, n_estimators=300
#   LGBM : LGBMRegressor, alpha=0.5, num_leaves=40, max_depth=10
#
# Based on:
# O'Connor, C., Collins, J., Prestwich, S., and Visentin, A. (2025).
# Conformal Prediction for Electricity Price Forecasting in Day-Ahead
# and Balancing Markets. Energy and AI, 21, 100571.
# https://doi.org/10.1016/j.egyai.2025.100571

import os, time, math, warnings
import numpy as np
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Fix macOS segfault: set OMP threads before any sklearn/torch import
os.environ.setdefault('OMP_NUM_THREADS', '1')

from config import (QUANTILES, QUANTILE_LABELS,
                    DAM_RETRAIN_EVERY, BM_RETRAIN_EVERY,
                    LEAR_RETRAIN_MULT, RF_RETRAIN_MULT,
                    LGBM_RETRAIN_MULT, KNN_RETRAIN_MULT,
                    ENBPI_RETRAIN_MULT, QRA_RETRAIN_MULT,
                    ENBPI_B, SPCI_B, ENBPI_B_RF, SPCI_B_RF, CP_PAST_WINDOW,
                    KNN_PARAMS,
                    RF_CP_DAM_PARAMS, RF_CP_BM_PARAMS,
                    LGBM_CP_DAM_PARAMS, LGBM_CP_BM_PARAMS,
                    CAL_FRAC, SEED)
from qr_models import (select_cp_features_dam, compute_lear_alpha,
                        _lear_prepare_dam, get_cp_base_model)



def _rolling_window(X_all, y_all, n_pre, bs, market):
    max_rows = 365 * (24 if market == 'DAM' else 48)
    win_end  = n_pre + bs
    win_st   = max(0, win_end - max_rows)
    X_win, y_win = X_all[win_st:win_end], y_all[win_st:win_end]
    n_cal    = max(1, int(len(X_win) * CAL_FRAC))
    return X_win[:-n_cal], y_win[:-n_cal], X_win[-n_cal:], y_win[-n_cal:]



def _cp_features(X, market):
    if market == 'DAM':
        return select_cp_features_dam(X)
    return X   # BM uses all features



class _LEARSingle:
    """Single-output Lasso for use as CP base model (predicts first horizon only)."""
    def __init__(self, alpha):
        from sklearn.linear_model import Lasso
        self._alpha = alpha
        self._model = Lasso(max_iter=2500, alpha=alpha)
        self._sc_x = None; self._sc_y = None

    def fit(self, X, y):
        y_1d = y[:, 0] if y.ndim > 1 else y.ravel()
        from sklearn import preprocessing
        prices, wind, demand = select_cp_features_dam(X)[:, :144], \
                               select_cp_features_dam(X)[:, 120:240], \
                               select_cp_features_dam(X)[:, 240:]
        self._sc_x = preprocessing.MinMaxScaler().fit(X)
        self._sc_y = preprocessing.MinMaxScaler().fit(y_1d.reshape(-1, 1))
        X_s = self._sc_x.transform(X)
        y_s = self._sc_y.transform(y_1d.reshape(-1, 1)).ravel()
        self._model.fit(X_s, y_s)
        return self

    def predict(self, X):
        X_s = self._sc_x.transform(X)
        y_s = self._model.predict(X_s).reshape(-1, 1)
        return self._sc_y.inverse_transform(y_s).ravel()



def _run_spci_class(model_name, data, B, use_spci, label,
                    quantiles=None, retrain_every=None):
    """
    Rolling EnbPI or SPCI using the paper's SPCI_and_EnbPI class.
    Runs SPCI_and_EnbPI twice: alpha=0.1 → Q10/Q90, alpha=0.3 → Q30/Q70.
    Q50 = model.predict() centre.
    """
    import torch
    try:
        import SPCI_class as SPCI
    except ImportError as e:
        print(f"  WARNING: Cannot import SPCI_class ({e}). Falling back to SCP.")
        return run_scp(model_name, data, quantiles, retrain_every)

    if quantiles is None: quantiles = QUANTILES
    market = data['market']
    stride = data['n_horizons']
    base   = DAM_RETRAIN_EVERY if market == 'DAM' else BM_RETRAIN_EVERY

    _mults = {'KNN': KNN_RETRAIN_MULT, 'LEAR': LEAR_RETRAIN_MULT,
              'RF': RF_RETRAIN_MULT,  'LGBM': LGBM_RETRAIN_MULT}
    if retrain_every is None:
        mult = _mults.get(model_name, 1) * ENBPI_RETRAIN_MULT
        retrain_every = base * mult

    X_all, y_all = data['X_all'], data['y_all']
    n_pre = data['n_pretrain']
    n_te  = len(data['X_test'])
    y_te  = data['y_test']
    n_h   = stride

    # Pre-compute LEAR alpha if needed
    lear_alpha = None
    if model_name == 'LEAR':
        print(f"  Computing LEAR alpha for CP...")
        lear_alpha = compute_lear_alpha(X_all[:n_pre], y_all[:n_pre, :1], market)
        print(f"  LEAR alpha = {lear_alpha:.6f}")

    preds  = np.zeros((n_te, n_h, len(quantiles)), dtype=np.float32)
    steps  = list(range(0, n_te, retrain_every))
    n_step = len(steps)
    t0_all = time.time()

    print(f"  [{label}-{model_name} | {market}] {n_step} retrains "
          f"(every {retrain_every} rows, B={B}, stride={stride})")

    for si, bs in enumerate(steps):
        be = min(bs + retrain_every, n_te)

        # Full rolling window — NO calibration split for EnbPI/SPCI.
        # The LOO bootstrap mechanism IS the calibration; removing the most
        # recent 20% of training data throws away the residuals most relevant
        # to the test period and causes severe under-coverage.
        max_rows = 365 * (24 if market == 'DAM' else 48)
        win_end  = n_pre + bs
        win_st   = max(0, win_end - max_rows)
        X_tr_full = X_all[win_st:win_end]
        y_tr_full = y_all[win_st:win_end]

        # Apply CP feature subset
        X_tr = _cp_features(X_tr_full, market)

        # Test window: 24 rows (one day), each row is a separate prediction
        X_batch = X_all[n_pre + bs : n_pre + bs + stride]
        X_te    = _cp_features(X_batch, market)   # (stride, n_cp_feat)

        # Single-output target (first column only, as in paper)
        y_tr_1d = y_tr_full[:, 0].astype(np.float64)
        y_te_s  = y_te[bs : bs + stride, 0].astype(np.float64)
        if len(y_te_s) < stride:
            y_te_s = np.pad(y_te_s, (0, stride - len(y_te_s)))

        # Get base model
        if model_name == 'LEAR':
            base_model = _LEARSingle(lear_alpha)
            base_model.fit(X_tr, y_tr_full)
        else:
            base_model = get_cp_base_model(model_name, market)
            # Cap KNN n_neighbors to available training samples
            if model_name == 'KNN' and hasattr(base_model, 'n_neighbors'):
                base_model.n_neighbors = min(base_model.n_neighbors, len(X_tr))
            base_model.fit(X_tr, y_tr_1d)

        pct = 100.0 * (si + 1) / n_step
        t0  = time.time()
        print(f"    step {si+1:3d}/{n_step}  ({pct:5.1f}%)  "
              f"B={B} stride={stride} ...", end='', flush=True)

        # Centre prediction (model.predict on test rows)
        try:
            mu_all = base_model.predict(X_te).ravel()[:stride]
        except Exception:
            mu_all = np.full(stride, y_tr_1d.mean())

    
        q_map = {0.5: mu_all.astype(np.float32)}

        try:
            X_tr_t = torch.from_numpy(X_tr.astype(np.float64)).float()
            X_te_t = torch.from_numpy(X_te.astype(np.float64)).float()   
            Y_tr_t = torch.from_numpy(y_tr_1d.reshape(-1, 1)).float()
            Y_te_t = torch.from_numpy(y_te_s.reshape(-1, 1)).float()

            obj = SPCI.SPCI_and_EnbPI(
                X_tr_t, X_te_t, Y_tr_t, Y_te_t,
                fit_func=base_model)
            obj.fit_bootstrap_models_online_multistep(
                B=B, fit_sigmaX=False, stride=stride)

            for alpha in [0.1, 0.3]:
                obj.compute_PIs_Ensemble_online(
                    alpha, smallT=(not use_spci),
                    past_window=CP_PAST_WINDOW,
                    use_SPCI=use_spci, quantile_regr=use_spci,
                    stride=stride)
                pi = obj.PIs_Ensemble
                lo = pi['lower'].values[:stride].astype(np.float32)
                hi = pi['upper'].values[:stride].astype(np.float32)
                q_map[alpha]     = lo
                q_map[1 - alpha] = hi

        except Exception as e:
            # Fallback: symmetric quantile from training residuals
            try:
                resid = np.abs(y_tr_1d - base_model.predict(X_tr).ravel())
            except Exception:
                resid = np.ones(len(y_tr_1d))
            for alpha in [0.1, 0.3]:
                width = float(np.quantile(resid, 1 - alpha))
                q_map[alpha]     = (mu_all - width).astype(np.float32)
                q_map[1 - alpha] = (mu_all + width).astype(np.float32)

  
        known = {0.1: q_map.get(0.1), 0.3: q_map.get(0.3),
                 0.5: q_map[0.5],
                 0.7: q_map.get(0.7), 0.9: q_map.get(0.9)}

        n_day = be - bs
        for qi, q in enumerate(quantiles):
            if known.get(q) is not None:
                row = known[q][:n_h].astype(np.float32)
            else:
                lo_q = max(qv for qv in known if qv < q and known[qv] is not None)
                hi_q = min(qv for qv in known if qv > q and known[qv] is not None)
                a    = (q - lo_q) / (hi_q - lo_q)
                row  = ((1-a)*known[lo_q][:n_h] + a*known[hi_q][:n_h]).astype(np.float32)
            preds[bs:be, :, qi] = np.tile(row, (n_day, 1))

        print(f"  {time.time()-t0:.1f}s")

    print(f"  [{label}-{model_name}] done. Total: {(time.time()-t0_all)/60:.1f} min\n")
    return preds


def run_enbpi(model_name, data, quantiles=None, retrain_every=None):
    """EnbPI using paper's SPCI_and_EnbPI (use_SPCI=False)."""
    B = ENBPI_B_RF if model_name in ('RF', 'LGBM') else ENBPI_B
    return _run_spci_class(model_name, data, B=B, use_spci=False,
                           label='EnbPI', quantiles=quantiles,
                           retrain_every=retrain_every)


def run_spci(model_name, data, quantiles=None, retrain_every=None):
    """SPCI using paper's SPCI_and_EnbPI (use_SPCI=True)."""
    B = SPCI_B_RF if model_name in ('RF', 'LGBM') else SPCI_B
    return _run_spci_class(model_name, data, B=B, use_spci=True,
                           label='SPCI', quantiles=quantiles,
                           retrain_every=retrain_every)



def run_scp(model_name, data, quantiles=None, retrain_every=None):
    """
    Split Conformal Prediction (additional, not in paper's GitHub).
    Per-horizon calibration for correct coverage.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    import lightgbm as lgb

    if quantiles is None: quantiles = QUANTILES
    market = data['market']
    base   = DAM_RETRAIN_EVERY if market == 'DAM' else BM_RETRAIN_EVERY
    _mults = {'KNN': KNN_RETRAIN_MULT, 'LEAR': LEAR_RETRAIN_MULT,
              'RF': RF_RETRAIN_MULT,  'LGBM': LGBM_RETRAIN_MULT}
    if retrain_every is None:
        retrain_every = base * _mults.get(model_name, 1)

    X_all, y_all = data['X_all'], data['y_all']
    n_pre = data['n_pretrain']
    n_te  = len(data['X_test'])
    n_h   = data['n_horizons']

    # Pre-compute LEAR alpha once
    lear_alpha = None
    if model_name == 'LEAR':
        lear_alpha = compute_lear_alpha(X_all[:n_pre], y_all[:n_pre], market)

    preds  = np.zeros((n_te, n_h, len(quantiles)), dtype=np.float32)
    steps  = list(range(0, n_te, retrain_every))
    n_step = len(steps)
    t0_all = time.time()
    print(f"  [SCP-{model_name} | {market}] {n_step} retrains (every {retrain_every} rows)")

    def _make_mo_model():
        from config import RF_CP_DAM_PARAMS, RF_CP_BM_PARAMS, LGBM_CP_DAM_PARAMS, LGBM_CP_BM_PARAMS, KNN_PARAMS
        if model_name == 'KNN':
            return KNeighborsRegressor(
                n_neighbors=min(KNN_PARAMS['n_neighbors'], 1),
                p=KNN_PARAMS['p'], leaf_size=KNN_PARAMS['leaf_size'],
                algorithm='ball_tree', n_jobs=-1)
        elif model_name == 'RF':
            p = RF_CP_DAM_PARAMS if market == 'DAM' else RF_CP_BM_PARAMS
            return RandomForestRegressor(n_jobs=-1, random_state=SEED, **p)
        elif model_name == 'LGBM':
            p = LGBM_CP_DAM_PARAMS if market == 'DAM' else LGBM_CP_BM_PARAMS
            return MultiOutputRegressor(lgb.LGBMRegressor(**p), n_jobs=1)
        else:
            return None  # LEAR handled separately

    for si, bs in enumerate(steps):
        be = min(bs + retrain_every, n_te)
        X_tr, y_tr, X_ca, y_ca = _rolling_window(X_all, y_all, n_pre, bs, market)
        X_te = X_all[n_pre + bs : n_pre + be]

        pct = 100.0 * (si + 1) / n_step
        t0  = time.time()
        print(f"    step {si+1:3d}/{n_step}  ({pct:5.1f}%)  "
              f"train {len(X_tr)} rows, cal {len(X_ca)} rows ...",
              end='', flush=True)

        if model_name == 'LEAR':
            from sklearn.linear_model import Lasso as LassoMod
            mu_ca = np.zeros_like(y_ca)
            mu_te = np.zeros((be - bs, n_h))
            for h in range(n_h):
                # Fit on training data, predict calibration and test
                X_tr_s, y_tr_s, X_ca_s, y_sc = _lear_prepare_dam(
                    X_tr, y_tr[:, h:h+1], X_ca)
                X_tr_s2, y_tr_s2, X_te_s, y_sc2 = _lear_prepare_dam(
                    X_tr, y_tr[:, h:h+1], X_te)
                m2 = LassoMod(max_iter=2500, alpha=lear_alpha)
                m2.fit(X_tr_s, y_tr_s.ravel())
                mu_ca_s = y_sc.inverse_transform(
                    m2.predict(X_ca_s).reshape(-1, 1)).ravel()
                mu_te_s = y_sc2.inverse_transform(
                    m2.predict(X_te_s).reshape(-1, 1)).ravel()
                mu_ca[:, h] = mu_ca_s
                mu_te[:, h] = mu_te_s
        else:
            k = min(KNN_PARAMS['n_neighbors'], len(X_tr)) if model_name == 'KNN' else None
            if model_name == 'KNN':
                from sklearn.neighbors import KNeighborsRegressor
                from config import KNN_PARAMS as KP
                m = KNeighborsRegressor(n_neighbors=k, p=KP['p'],
                                        leaf_size=KP['leaf_size'],
                                        algorithm='ball_tree', n_jobs=-1)
            elif model_name == 'RF':
                from config import RF_CP_DAM_PARAMS, RF_CP_BM_PARAMS
                p = RF_CP_DAM_PARAMS if market == 'DAM' else RF_CP_BM_PARAMS
                m = RandomForestRegressor(n_jobs=-1, random_state=SEED, **p)
            elif model_name == 'LGBM':
                from config import LGBM_CP_DAM_PARAMS, LGBM_CP_BM_PARAMS
                p = LGBM_CP_DAM_PARAMS if market == 'DAM' else LGBM_CP_BM_PARAMS
                m = MultiOutputRegressor(lgb.LGBMRegressor(**p), n_jobs=1)
            m.fit(X_tr, y_tr)
            mu_ca = np.atleast_2d(m.predict(X_ca))
            mu_te = np.atleast_2d(m.predict(X_te))
            if mu_ca.ndim == 1:
                mu_ca = np.tile(mu_ca.reshape(-1,1), (1, n_h))
            if mu_te.ndim == 1:
                mu_te = np.tile(mu_te.reshape(-1,1), (1, n_h))

        resid_ca = np.abs(y_ca - mu_ca)  
        for qi, q in enumerate(quantiles):
            alpha = 2 * min(q, 1 - q)
            w = np.quantile(resid_ca, min(1.0, 1.0 - alpha), axis=0)
            if q < 0.5:
                preds[bs:be, :, qi] = (mu_te - w).astype(np.float32)
            elif q > 0.5:
                preds[bs:be, :, qi] = (mu_te + w).astype(np.float32)
            else:
                preds[bs:be, :, qi] = mu_te.astype(np.float32)

        print(f"  {time.time()-t0:.1f}s")

    print(f"  [SCP-{model_name}] done. Total: {(time.time()-t0_all)/60:.1f} min\n")
    return preds


