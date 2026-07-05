# Asymmetric Volatility-Aware Conformal PID Control (AV-C-PID)
#
# This file implements AV-C-PID, a novel extension of the conformal PID control
# framework from Angelopoulos et al. (2023) for electricity price forecasting.
# The method adaptively widens prediction intervals during volatile periods and
# calibrates upper and lower bounds independently to handle the asymmetric
# error distribution of electricity prices.
#
# Based on:
# Angelopoulos, A., Bates, S., Malik, J., and Jordan, M. (2023).
# Conformal PID Control for Time Series Prediction.
# Advances in Neural Information Processing Systems (NeurIPS 2023).
# https://arxiv.org/abs/2307.16895

import time, math, warnings
import numpy as np
warnings.filterwarnings('ignore')
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
except ImportError:
    pass

from config import (QUANTILES, QUANTILE_LABELS,
                    DAM_RETRAIN_EVERY, BM_RETRAIN_EVERY, SEED)


LAMBDA   = 0.94   # EWMA decay
ETA      = 0.1    # proportional learning rate base
CSAT     = 5.0    # saturation constant
KI       = 10.0   # integrator gain
AHEAD    = 1      # one-step ahead


T_BURNIN          = 30 * 24    # 30 days before Theta scorecaster activates
ERR_WINDOW        = 7 * 24     # 7-day trailing window for the I-control sum
REFRESH_INTERVAL  = 30 * 24    # refresh σ ceiling/floor roughly monthly
REFRESH_WINDOW    = 180 * 24   # use last 6 months of residuals for refresh
REFRESH_MIN_OBS   = 60 * 24    # don't refresh until this many hours collected


Q_FLOOR_RATIO    = 0.20   
Q_FLOOR_ABS_MIN  = 0.5
Q_CEIL_RATIO     = 3.0   
Q_CEIL_ABS_MAX   = 4.0

Q_RATE_LIMIT     = 1.5 ** (1/24)  


ALPHA_BIAS = 0.40   
                    
BETA_BIAS  = 0.10   

THETA_HISTORY_CAP = 365 * 24  


def _saturation(x, t, Csat=CSAT, KI=KI):
    if KI == 0 or t <= 0:
        return 0.0
    arg = x * math.log(t + 1) / (Csat * (t + 1))
    if   arg >=  math.pi / 2: return  KI * 1e9
    elif arg <= -math.pi / 2: return -KI * 1e9
    return KI * math.tan(arg)


def _theta_forecast(q_history, seasonal_period=24):
    n = len(q_history)
    if n < 4:
        return q_history[-1] if n > 0 else 0.0
    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel
        model = ThetaModel(
            np.nan_to_num(q_history).astype(float),
            period=max(2, min(seasonal_period, n // 2)),
            use_mle=False
        ).fit(disp=False)
        return float(model.forecast(1)[0])
    except Exception:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(
                np.nan_to_num(q_history).astype(float),
                trend='add', initialization_method='estimated'
            ).fit(optimized=True, disp=False)
            return float(model.forecast(1)[0])
        except Exception:
            return float(q_history[-1])



def _get_training_residuals(model_name, data):

    from qr_models import (fit_predict_knn, fit_predict_lear,
                            fit_predict_rf, fit_predict_lgbm,
                            compute_lear_alpha)
    market  = data['market']
    X_all   = data['X_all']
    y_all   = data['y_all']
    n_pre   = data['n_pretrain']
    n_h     = data['n_horizons']
    # LEAR uses a shorter window due to computation time 
    from config import LEAR_MAX_TRAIN_ROWS, LEAR_BM_MAX_TRAIN_ROWS
    if model_name == 'LEAR':
        max_rows = LEAR_MAX_TRAIN_ROWS if market == 'DAM' else LEAR_BM_MAX_TRAIN_ROWS
    else:
        max_rows = 365 * (24 if market == 'DAM' else 48)

    win_st  = max(0, n_pre - max_rows)
    X_tr    = X_all[win_st:n_pre]
    y_tr    = y_all[win_st:n_pre]
    n_split = int(len(X_tr) * 0.8)
    X_fit, y_fit = X_tr[:n_split], y_tr[:n_split]
    X_cal, y_cal = X_tr[n_split:], y_tr[n_split:]

    try:
        if model_name == 'LEAR':
            alpha = compute_lear_alpha(X_fit, y_fit, market)
            p = fit_predict_lear(X_fit, y_fit, X_cal, alpha, market)
        elif model_name == 'KNN':
            p = fit_predict_knn(X_fit, y_fit, X_cal, market)
        elif model_name == 'RF':
            p = fit_predict_rf(X_fit, y_fit, X_cal, market)
        elif model_name == 'LGBM':
            p = fit_predict_lgbm(X_fit, y_fit, X_cal, market)
        else:
            raise ValueError(model_name)
        mu_tr    = p[:, :, 2].astype(np.float64)   # Q50
        y_tr_cal = y_cal.astype(np.float64)
    except Exception as e:
        print(f"  WARNING: training residual init failed ({e}), using variance")
        mu_tr    = y_cal.mean(axis=0, keepdims=True).repeat(len(y_cal), 0)
        y_tr_cal = y_cal.astype(np.float64)

    return mu_tr, y_tr_cal



def _avcpid_chronological(mu_flat, y_flat, mu_tr_flat, y_tr_flat, alpha,
                          seasonal_period=24, progress_label="",
                          progress_every=500):
    
    n_te = len(mu_flat)

    e_tr_raw   = (y_tr_flat - mu_tr_flat).astype(np.float64)
    bias_level = float(np.mean(e_tr_raw))
    bias_trend = 0.0

    e_tr     = e_tr_raw - bias_level
    var_init = max(float(np.var(e_tr)), 1e-6)
    sigma2_min = var_init / 25.0
    sigma2_max = 4.0 * var_init
    sigma2 = var_init

    sigma_init = math.sqrt(var_init)
    s_U_tr = np.maximum(e_tr, 0.0) / sigma_init
    s_L_tr = np.maximum(-e_tr, 0.0) / sigma_init
    q_U = float(np.quantile(s_U_tr, 1 - alpha))
    q_L = float(np.quantile(s_L_tr, 1 - alpha))

    Q_MIN_U = max(q_U * Q_FLOOR_RATIO, Q_FLOOR_ABS_MIN)
    Q_MIN_L = max(q_L * Q_FLOOR_RATIO, Q_FLOOR_ABS_MIN)
    Q_MAX_U = min(q_U * Q_CEIL_RATIO, Q_CEIL_ABS_MAX)
    Q_MAX_L = min(q_L * Q_CEIL_RATIO, Q_CEIL_ABS_MAX)
    q_U = max(min(q_U, Q_MAX_U), Q_MIN_U)
    q_L = max(min(q_L, Q_MAX_L), Q_MIN_L)

    qt_U = q_U;  qt_L = q_L
    err_hist_U = [];  err_hist_L = []
    q_U_hist   = [q_U];  q_L_hist = [q_L]

    lo = np.zeros(n_te, dtype=np.float32)
    hi = np.zeros(n_te, dtype=np.float32)
    resid_buffer = []

   
    LAG = seasonal_period   # 24 for DAM, 16 for BM

    e_raw_buffer = {}

    t0 = time.time()
    for t in range(n_te):
        sigma = math.sqrt(max(sigma2, 1e-6))

        predicted_bias = bias_level + bias_trend
        center_t = mu_flat[t] + predicted_bias

        lo[t] = center_t - q_L * sigma
        hi[t] = center_t + q_U * sigma


        e_t      = float(y_flat[t]) - center_t
        s_U_t    = max(e_t, 0.0) / sigma
        s_L_t    = max(-e_t, 0.0) / sigma
        covered_U = (s_U_t <= q_U)
        covered_L = (s_L_t <= q_L)

        e_raw_buffer[t] = float(y_flat[t]) - float(mu_flat[t])

        if t >= LAG:
           
            raw_lag   = e_raw_buffer[t - LAG]
            
            e_lag     = raw_lag - predicted_bias

            # PID coverage update using lagged observation
            s_U_lag   = max(e_lag, 0.0) / sigma
            s_L_lag   = max(-e_lag, 0.0) / sigma

            err_U = 0.0 if s_U_lag <= q_U else 1.0
            err_L = 0.0 if s_L_lag <= q_L else 1.0
            err_hist_U.append(err_U - alpha)
            err_hist_L.append(err_L - alpha)
            err_sum_U = sum(err_hist_U[-ERR_WINDOW:])
            err_sum_L = sum(err_hist_L[-ERR_WINDOW:])

            win_st = max(0, t + 1 - T_BURNIN)
            seen_U = np.maximum(mu_flat[win_st:t+1] - y_flat[win_st:t+1], 0.0)
            seen_L = np.maximum(y_flat[win_st:t+1] - mu_flat[win_st:t+1], 0.0)
            lr_U = ETA * (float(np.ptp(seen_U)) / sigma + 1e-8)
            lr_L = ETA * (float(np.ptp(seen_L)) / sigma + 1e-8)

            grad_U = alpha if s_U_lag <= q_U else -(1 - alpha)
            grad_L = alpha if s_L_lag <= q_L else -(1 - alpha)
            qt_U_new = qt_U - lr_U * grad_U
            qt_L_new = qt_L - lr_L * grad_L

            int_U = _saturation(err_sum_U, t + 1)
            int_L = _saturation(err_sum_L, t + 1)

            if t >= T_BURNIN and len(q_U_hist) >= 4:
                sc_U = _theta_forecast(
                    np.array(q_U_hist[-min(THETA_HISTORY_CAP, len(q_U_hist)):]),
                    seasonal_period)
                sc_L = _theta_forecast(
                    np.array(q_L_hist[-min(THETA_HISTORY_CAP, len(q_L_hist)):]),
                    seasonal_period)
            else:
                sc_U = sc_L = 0.0

            q_U_raw = qt_U_new + int_U + sc_U
            q_L_raw = qt_L_new + int_L + sc_L
            q_U_max_step = max(q_U * Q_RATE_LIMIT, Q_MIN_U * 2)
            q_L_max_step = max(q_L * Q_RATE_LIMIT, Q_MIN_L * 2)
            q_U = max(min(q_U_raw, min(q_U_max_step, Q_MAX_U)), Q_MIN_U)
            q_L = max(min(q_L_raw, min(q_L_max_step, Q_MAX_L)), Q_MIN_L)
            qt_U = qt_U_new;  qt_L = qt_L_new
            q_U_hist.append(q_U);  q_L_hist.append(q_L)

            # sigma update using lagged centered residual
            sigma2 = max(min(LAMBDA * sigma2 + (1 - LAMBDA) * e_lag ** 2,
                             sigma2_max), sigma2_min)

            # Periodic ceiling/floor refresh
            resid_buffer.append(e_lag)
            if len(resid_buffer) > REFRESH_WINDOW:
                resid_buffer.pop(0)
            if (t + 1) >= REFRESH_MIN_OBS and (t + 1) % REFRESH_INTERVAL == 0:
                fresh_var = max(float(np.var(resid_buffer)), 1e-6)
                sigma2_min = fresh_var / 25.0
                sigma2_max = 4.0 * fresh_var
                sigma2 = max(min(sigma2, sigma2_max), sigma2_min)

            day_resids = [e_raw_buffer[t - h]
                          for h in range(1, LAG + 1)
                          if (t - h) in e_raw_buffer]
            update_signal = float(np.mean(day_resids)) if day_resids else raw_lag
            new_level = ALPHA_BIAS * update_signal + (1 - ALPHA_BIAS) * (bias_level + bias_trend)
            new_trend = BETA_BIAS * (new_level - bias_level) + (1 - BETA_BIAS) * bias_trend
            bias_level, bias_trend = new_level, new_trend

            if (t - LAG) in e_raw_buffer:
                del e_raw_buffer[t - LAG]

        if progress_every and (t + 1) % progress_every == 0:
            pct = 100 * (t + 1) / n_te
            elapsed = time.time() - t0
            eta = elapsed / (t + 1) * (n_te - t - 1)
            print(f"    {progress_label} step {t+1:6d}/{n_te} "
                  f"({pct:5.1f}%)  elapsed={elapsed:6.1f}s  eta={eta:6.1f}s")

    return lo, hi


def run_avcpid(model_name, data, quantiles=None, retrain_every=None,
               qr_preds=None):

    if quantiles is None:
        quantiles = QUANTILES
    market = data['market']
    n_h    = data['n_horizons']
    y_te   = data['y_test'].astype(np.float64)   # (n_test, n_h)
    n_te   = len(y_te)
    seasonal_period  = 24 if market == 'DAM' else 16
    from config import DAM_RETRAIN_EVERY, BM_RETRAIN_EVERY

    print(f"  [AV-C-PID-{model_name} | {market}] "
          f"base=Q50({model_name}), scorecaster=Theta, "
          f"λ={LAMBDA}, Csat={CSAT}, KI={KI}, update=hourly")
    t0 = time.time()

    if qr_preds is not None:
        retrain = DAM_RETRAIN_EVERY if market == 'DAM' else BM_RETRAIN_EVERY
        qr_sub = qr_preds[::retrain]           # (n_days, n_h, 5)
        n_days_qr = len(qr_sub)
        n_days_te = n_te // retrain

        if n_days_qr != n_days_te:
            print(f"  WARNING: QR predictions have {n_days_qr} days but test "
                  f"has {n_days_te} days. Recomputing QR Q50...")
            from qr_models import run_qr
            qr_p  = run_qr(model_name, data, quantiles=quantiles)
            mu_te = qr_p[:, :, 2].astype(np.float64)
        else:
            print(f"  Using pre-computed {model_name} QR Q50 as base forecasts.")
            mu_full = np.repeat(qr_sub[:, :, 2], retrain, axis=0)[:n_te]
            mu_te   = mu_full.astype(np.float64)
    else:
        print(f"  Computing {model_name} QR Q50 for base forecasts...")
        from qr_models import run_qr
        qr_p  = run_qr(model_name, data, quantiles=quantiles)
        mu_te = qr_p[:, :, 2].astype(np.float64)

    print(f"  Fitting {model_name} on training data for residual initialization...")
    mu_tr, y_tr_cal = _get_training_residuals(model_name, data)
    mu_tr_flat = mu_tr.flatten().astype(np.float64)
    y_tr_flat  = y_tr_cal.flatten().astype(np.float64)

    mu_flat = mu_te.flatten()
    y_flat  = y_te.flatten()
    n_flat  = len(mu_flat)

    print(f"  Running chronological hourly AV-C-PID "
          f"({n_flat} hourly steps × 2 alphas)...")

    preds = np.zeros((n_te, n_h, len(quantiles)), dtype=np.float32)
    q50_idx = quantiles.index(0.5) if 0.5 in quantiles else 2
    preds[:, :, q50_idx] = mu_te.astype(np.float32)

    progress_every = max(500, n_flat // 20)   
    print(f"  [alpha=0.1] computing 80% interval bounds...")
    lo10, hi90 = _avcpid_chronological(
        mu_flat, y_flat, mu_tr_flat, y_tr_flat, alpha=0.1,
        seasonal_period=seasonal_period,
        progress_label=f"[{model_name}|a=0.1]", progress_every=progress_every)

    print(f"  [alpha=0.3] computing 40% interval bounds...")
    lo30, hi70 = _avcpid_chronological(
        mu_flat, y_flat, mu_tr_flat, y_tr_flat, alpha=0.3,
        seasonal_period=seasonal_period,
        progress_label=f"[{model_name}|a=0.3]", progress_every=progress_every)

    lo10 = lo10.reshape(n_te, n_h)
    hi90 = hi90.reshape(n_te, n_h)
    lo30 = lo30.reshape(n_te, n_h)
    hi70 = hi70.reshape(n_te, n_h)

    q_idx = {q: i for i, q in enumerate(quantiles)}
    if 0.1 in q_idx: preds[:, :, q_idx[0.1]] = lo10
    if 0.3 in q_idx: preds[:, :, q_idx[0.3]] = lo30
    if 0.7 in q_idx: preds[:, :, q_idx[0.7]] = hi70
    if 0.9 in q_idx: preds[:, :, q_idx[0.9]] = hi90

    for qi in range(1, len(quantiles)):
        preds[:, :, qi] = np.maximum(preds[:, :, qi], preds[:, :, qi-1])

    print(f"  [AV-C-PID-{model_name}] done. "
          f"Total: {(time.time()-t0)/60:.1f} min\n")
    return preds



def run_pid_q_ens(qr_preds, enbpi_preds, spci_preds, avcpid_preds):
    """PID-Q-Ens: equal-weight average of QR + EnbPI + SPCI + AV-C-PID."""
    return ((qr_preds.astype(np.float32)
             + enbpi_preds.astype(np.float32)
             + spci_preds.astype(np.float32)
             + avcpid_preds.astype(np.float32)) / 4.0)
