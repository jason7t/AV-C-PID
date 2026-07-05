"""
qra_methods.py  —  QRA-R and QRA-CP (additional methods, not in paper's GitHub).

QRA-R:  Quantile Regression Averaging combining QR predictions from sub-windows.
QRA-CP: Quantile Regression Averaging combining SCP, EnbPI, SPCI predictions.
Q-Ens:  Simple average of QR + EnbPI + SPCI.
"""
import time, warnings
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore')
from config import (QUANTILES, QUANTILE_LABELS, QRA_RETRAIN_MULT,
                    DAM_RETRAIN_EVERY, BM_RETRAIN_EVERY,
                    LEAR_RETRAIN_MULT, RF_RETRAIN_MULT,
                    LGBM_RETRAIN_MULT, KNN_RETRAIN_MULT, CAL_FRAC, SEED)
from qr_models import (fit_predict_lear, fit_predict_rf, fit_predict_lgbm,
                        fit_predict_knn, compute_lear_alpha)
from cp_methods import _rolling_window, run_scp, run_enbpi, run_spci


_MULTS = {'KNN': KNN_RETRAIN_MULT, 'LEAR': LEAR_RETRAIN_MULT,
          'RF': RF_RETRAIN_MULT,   'LGBM': LGBM_RETRAIN_MULT}


def run_qra_r(model_name, data, quantiles=None, retrain_every=None):
    """
    QRA-R: train QR on 3 sub-windows, combine with linear Quantile Regression.
    Windows: [full], [last 2/3], [last 1/3].
    """
    if quantiles is None: quantiles = QUANTILES
    market = data['market']
    base   = DAM_RETRAIN_EVERY if market == 'DAM' else BM_RETRAIN_EVERY
    if retrain_every is None:
        retrain_every = base * _MULTS.get(model_name, 1)

    X_all, y_all = data['X_all'], data['y_all']
    n_pre = data['n_pretrain']
    n_te  = len(data['X_test'])
    n_h   = data['n_horizons']

    lear_alpha = None
    if model_name == 'LEAR':
        lear_alpha = compute_lear_alpha(X_all[:n_pre], y_all[:n_pre], market)

    _fp = {'KNN': fit_predict_knn, 'LEAR': fit_predict_lear,
           'RF': fit_predict_rf,   'LGBM': fit_predict_lgbm}
    fp = _fp[model_name]

    preds  = np.zeros((n_te, n_h, len(quantiles)), dtype=np.float32)
    steps  = list(range(0, n_te, retrain_every))
    n_step = len(steps)
    t0_all = time.time()
    print(f"  [QRA-R-{model_name} | {market}] {n_step} retrains")

    for si, bs in enumerate(steps):
        be = min(bs + retrain_every, n_te)
        X_tr, y_tr, X_ca, y_ca = _rolling_window(X_all, y_all, n_pre, bs, market)
        X_te = X_all[n_pre + bs : n_pre + be]

        n_tr = len(X_tr)
        # 3 sub-windows
        slices = [slice(0, n_tr), slice(n_tr//3, n_tr), slice(2*n_tr//3, n_tr)]
        cal_preds = np.zeros((len(X_ca), n_h, len(quantiles), 3), dtype=np.float32)
        te_preds  = np.zeros((be-bs, n_h, len(quantiles), 3), dtype=np.float32)

        pct = 100.0 * (si + 1) / n_step
        t0  = time.time()
        print(f"    step {si+1:3d}/{n_step}  ({pct:5.1f}%)  "
              f"3 sub-windows ...", end='', flush=True)

        for k, sl in enumerate(slices):
            X_sub, y_sub = X_tr[sl], y_tr[sl]
            if len(X_sub) < 10: X_sub, y_sub = X_tr, y_tr
            if model_name == 'LEAR':
                # Previously used KNN as a "proxy" for LEAR here, since the
                # LP solver was too slow for 3 sub-window fits per retrain
                # step. BUG: this meant LEAR's QRA-R results were actually
                # KNN's predictions relabeled — identical to KNN's QRA-R
                # output (confirmed via bit-for-bit duplicate values in
                # BM_metrics.csv). Fixed by actually fitting LEAR.
                #
                # Two optimizations vs a naive fix, both benchmarked:
                # 1) Aggressive row caps — measured fit times of 102s/44s/25s
                #    at 720/480/240 rows respectively. Even 240 rows is too
                #    slow given this runs 3x per retrain step x 365 steps.
                #    Capped further to [150,100,50] rows (still preserves
                #    the 3:2:1 relative-window-length ratio QRA-R needs).
                # 2) Fit ONCE per slice instead of twice — the original code
                #    called fit_predict_lear separately for X_ca and X_te,
                #    each refitting the model from scratch (wasteful, 2x
                #    cost for no reason). Now concatenates X_ca+X_te into a
                #    single prediction call and splits the output.
                LEAR_QRA_CAPS = [150, 100, 50]   # full : 2/3 : 1/3 ratio
                cap_k = LEAR_QRA_CAPS[k]
                X_sub_l, y_sub_l = X_sub, y_sub
                if len(X_sub_l) > cap_k:
                    X_sub_l = X_sub_l[-cap_k:]
                    y_sub_l = y_sub_l[-cap_k:]
                sub_alpha = compute_lear_alpha(X_sub_l, y_sub_l, market)
                n_ca = len(X_ca)
                X_combined = np.concatenate([X_ca, X_te], axis=0)
                p_combined = fit_predict_lear(X_sub_l, y_sub_l, X_combined, sub_alpha, market)
                cal_preds[:,:,:,k] = p_combined[:n_ca]
                te_preds[:,:,:,k]  = p_combined[n_ca:]
            elif model_name == 'LGBM':
                # Reduce n_estimators AND row count for sub-window fits —
                # these are only QRA combiner features, not the final
                # output. The "full" sub-window (slice 0) was previously
                # the entire 8760-row training set even at n_estimators=30,
                # since LGBM fit time scales with row count regardless of
                # tree count (histogram-building cost). Capping every
                # sub-window to at most 1000 rows bounds runtime
                # consistently while preserving each window's *relative*
                # recency (most recent rows kept).
                import lightgbm as lgb
                from sklearn.multioutput import MultiOutputRegressor
                from config import LGBM_QR_DAM_PARAMS, LGBM_QR_BM_PARAMS
                p = (LGBM_QR_DAM_PARAMS if market == 'DAM'
                     else LGBM_QR_BM_PARAMS).copy()
                p['n_estimators'] = 30
                # Cap PER-SLICE, preserving each window's relative length —
                # a single flat cap collapses all 3 windows into the SAME
                # data once the cap is smaller than the shortest original
                # slice (2/3 of n_tr ≈ 2920 rows), destroying the diversity
                # QRA-R's "3 sub-windows" relies on. These caps keep the
                # original 3:2:1 ratio (full : 2/3 : 1/3) while bounding
                # absolute cost.
                LGBM_QRA_CAPS = [300, 200, 100]   # one per slice index k
                # Benchmarked: gives ~11x speedup vs uncapped (37h -> ~3.4h
                # for 365 days) while preserving 3 genuinely different
                # window lengths (3:2:1 ratio kept) for the QRA combiner.
                cap_k = LGBM_QRA_CAPS[k]
                if len(X_sub) > cap_k:
                    X_sub = X_sub[-cap_k:]
                    y_sub = y_sub[-cap_k:]
                def _lgbm_sub(X_f, y_f, X_p):
                    from config import QUANTILES
                    import numpy as np
                    n_h = y_f.shape[1]
                    out = np.zeros((len(X_p), n_h, len(QUANTILES)), dtype=np.float32)
                    for qi, q in enumerate(QUANTILES):
                        m = MultiOutputRegressor(
                            lgb.LGBMRegressor(**{**p, 'alpha': q}), n_jobs=1)
                        m.fit(X_f, y_f)
                        out[:, :, qi] = m.predict(X_p)
                    return out
                cal_preds[:,:,:,k] = _lgbm_sub(X_sub, y_sub, X_ca)
                te_preds[:,:,:,k]  = _lgbm_sub(X_sub, y_sub, X_te)
            else:
                cal_preds[:,:,:,k] = fp(X_sub, y_sub, X_ca, market)
                te_preds[:,:,:,k]  = fp(X_sub, y_sub, X_te, market)

        # Fit QRA combiner: for each horizon and quantile, regress y_ca on 3 predictions
        for qi, q in enumerate(quantiles):
            for h in range(n_h):
                X_qra = cal_preds[:, h, qi, :]   # (n_cal, 3)
                y_qra = y_ca[:, h]
                qr = QuantileRegressor(quantile=q, alpha=0.01, solver='highs-ds')
                try:
                    qr.fit(X_qra, y_qra)
                    preds[bs:be, h, qi] = qr.predict(te_preds[:, h, qi, :])
                except Exception:
                    preds[bs:be, h, qi] = te_preds[:, h, qi, 0]

        print(f"  {time.time()-t0:.1f}s")

    print(f"  [QRA-R-{model_name}] done. Total: {(time.time()-t0_all)/60:.1f} min\n")
    return preds


def run_qra_cp(model_name, data, quantiles=None, retrain_every=None,
               scp_preds=None, enbpi_preds=None, spci_preds=None):
    """
    QRA-CP: combine predictions from SCP, EnbPI, SPCI with linear QRA combiner.
    Pass pre-computed scp_preds/enbpi_preds/spci_preds to avoid recomputing.
    """
    if quantiles is None: quantiles = QUANTILES
    market = data['market']
    base   = DAM_RETRAIN_EVERY if market == 'DAM' else BM_RETRAIN_EVERY
    if retrain_every is None:
        retrain_every = base * _MULTS.get(model_name, 1)

    n_te = len(data['X_test'])
    n_h  = data['n_horizons']

    print(f"  [QRA-CP-{model_name} | {market}] computing component forecasts...")
    scp_p   = scp_preds   if scp_preds   is not None else run_scp(model_name, data, quantiles, retrain_every)
    enbpi_p = enbpi_preds if enbpi_preds is not None else run_enbpi(model_name, data, quantiles, retrain_every)
    spci_p  = spci_preds  if spci_preds  is not None else run_spci(model_name, data, quantiles, retrain_every)

    y_te = data['y_test']
    n_cal = max(1, int(n_te * CAL_FRAC))
    n_tr_te = n_te - n_cal

    preds = np.zeros((n_te, n_h, len(quantiles)), dtype=np.float32)
    print(f"  [QRA-CP-{model_name}] fitting QRA combiner...")

    for qi, q in enumerate(quantiles):
        for h in range(n_h):
            X_qra_tr = np.column_stack([scp_p[:n_tr_te, h, qi],
                                         enbpi_p[:n_tr_te, h, qi],
                                         spci_p[:n_tr_te, h, qi]])
            y_qra_tr = y_te[:n_tr_te, h]
            qr = QuantileRegressor(quantile=q, alpha=0.01, solver='highs-ds')
            try:
                qr.fit(X_qra_tr, y_qra_tr)
                X_qra_te = np.column_stack([scp_p[:, h, qi],
                                             enbpi_p[:, h, qi],
                                             spci_p[:, h, qi]])
                preds[:, h, qi] = qr.predict(X_qra_te)
            except Exception:
                preds[:, h, qi] = (scp_p[:, h, qi] + enbpi_p[:, h, qi] +
                                   spci_p[:, h, qi]) / 3

    print(f"  [QRA-CP-{model_name}] done.\n")
    return preds


def run_q_ens(qr_preds, enbpi_preds, spci_preds):
    """Q-Ens: simple average of QR + EnbPI + SPCI."""
    return ((qr_preds.astype(np.float32)
             + enbpi_preds.astype(np.float32)
             + spci_preds.astype(np.float32)) / 3.0)
