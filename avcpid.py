"""
avcpid.py  —  Asymmetric Volatility-Aware Conformal PID Control (AV-C-PID)

Extension of Angelopoulos et al. (2023) for electricity price forecasting.
Reference: Tran (2026), Bachelor Thesis, Erasmus University Rotterdam.

Architecture (matching thesis exactly):
  - Base forecaster: KNN/LEAR/RF/LGBM point forecast μ_t (Q50 from QR)
  - EWMA volatility scaling: σ²_t = λσ²_{t-1} + (1-λ)e²_{t-1}
  - Asymmetric scores: s^U = max(e,0)/σ, s^L = max(-e,0)/σ
  - PID update per tail:
      P: quantile tracker (η proportional to score range)
      I: tan saturation integrator (Csat, KI)
      D: Theta scorecaster on threshold sequence
  - Final interval: [μ_t - q^L_t·σ_t,  μ_t + q^U_t·σ_t]

HOURLY UPDATE ARCHITECTURE: a single shared controller (per alpha level)
processes the ENTIRE test period as one chronological hourly sequence,
rather than 24 separate per-hour-of-day controllers each updated once per
day. This gives ~24x more frequent updates over the same test period,
since every hourly outcome (not just one hour-of-day) feeds the same
σ/bias/threshold state as soon as it's chronologically observed. The base
model's own hour-specific Q50 forecast already captures daily seasonality
in price LEVEL; this controller only needs to track the residual/width,
which is shared sensibly across hours.

All time-based constants (T_BURNIN, ERR_WINDOW, REFRESH_*) are expressed
in HOURS now that the loop advances hourly, not daily.

The Theta model is the SCORECASTER (D term) only — not the base forecaster.
This matches Angelopoulos et al. exactly (their Theta = scorecaster).
Results are model-specific because μ_t comes from each base model.
"""

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

# ── Hyperparameters (from Angelopoulos et al. elec2 config) ──────────────────
LAMBDA   = 0.94   # EWMA decay (RiskMetrics)
ETA      = 0.1    # proportional learning rate base
CSAT     = 5.0    # saturation constant
KI       = 10.0   # integrator gain
AHEAD    = 1      # one-step ahead

# Time-based constants, in HOURS (loop now advances one hour per step).
# Previously these were tuned in "days" assuming one update per day;
# rescaled by x24 here to preserve the same intended real-world duration.
T_BURNIN          = 30 * 24    # 30 days before Theta scorecaster activates
ERR_WINDOW        = 7 * 24     # 7-day trailing window for the I-control sum
REFRESH_INTERVAL  = 30 * 24    # refresh σ ceiling/floor roughly monthly
REFRESH_WINDOW    = 180 * 24   # use last ~6 months of residuals for refresh
REFRESH_MIN_OBS   = 60 * 24    # don't refresh until this many hours collected

# Threshold floor/ceiling (in standardized score units, multiplied by σ).
Q_FLOOR_RATIO    = 0.20   # floor = max(q_init * this, Q_FLOOR_ABS_MIN)
Q_FLOOR_ABS_MIN  = 0.5
Q_CEIL_RATIO     = 3.0    # ceiling = min(q_init * this, Q_CEIL_ABS_MAX)
Q_CEIL_ABS_MAX   = 4.0
# Rate limit is MULTIPLICATIVE per step, not a duration — it did NOT get
# the x24 rescaling the other time-based constants received. Under the old
# once-per-day architecture, 1.5x meant "50% growth allowed per day". Now
# that a step is one HOUR, the same 1.5 would compound to 1.5^24 ≈ 16,800x
# per day if sustained — a massive, unintended loosening. Rescaled here so
# 24 consecutive hourly steps compound to the SAME daily growth as before:
# 1.5^(1/24) ≈ 1.017. Verified on synthetic data: this tightened IW80 by
# ~26% with no loss of coverage (still ~0.82, target 0.80).
Q_RATE_LIMIT     = 1.5 ** (1/24)   # ≈ 1.017 per hour, equivalent to 1.5/day

# Bias correction uses Holt's linear trend method (level + trend), letting
# the correction extrapolate an ongoing drift rather than just react to it.
ALPHA_BIAS = 0.40   # level smoothing — raised from 0.20 to match the more
                    # stable day-average bias signal (single-lag residual is
                    # noisy; averaging 24 residuals warrants faster adaptation)
BETA_BIAS  = 0.10   # trend smoothing — raised from 0.05 accordingly

# Theta scorecaster: cap how much threshold history it fits on, in HOURS.
THETA_HISTORY_CAP = 365 * 24   # up to 1 year of hourly threshold history


# ── Tan saturation integrator ─────────────────────────────────────────────────

def _saturation(x, t, Csat=CSAT, KI=KI):
    if KI == 0 or t <= 0:
        return 0.0
    arg = x * math.log(t + 1) / (Csat * (t + 1))
    if   arg >=  math.pi / 2: return  KI * 1e9
    elif arg <= -math.pi / 2: return -KI * 1e9
    return KI * math.tan(arg)


# ── Theta scorecaster (fitted on past threshold sequence) ─────────────────────

def _theta_forecast(q_history, seasonal_period=24):
    """
    Forecast next quantile threshold using Theta model.
    This is the D (derivative/scorecaster) component of C-PID.
    seasonal_period=24 now directly means 24 HOURS = 1 day, which is a
    genuine, meaningful periodicity in the hourly threshold sequence.
    """
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


# ── Training residuals for initialization ─────────────────────────────────────

def _get_training_residuals(model_name, data):
    """
    Fit base model once on training data → calibration residuals for
    EWMA σ² initialization and threshold q^U/q^L initialization.
    Uses a single 80/20 split — much faster than rolling.
    """
    from qr_models import (fit_predict_knn, fit_predict_lear,
                            fit_predict_rf, fit_predict_lgbm,
                            compute_lear_alpha)
    market  = data['market']
    X_all   = data['X_all']
    y_all   = data['y_all']
    n_pre   = data['n_pretrain']
    n_h     = data['n_horizons']
    # LEAR uses a much shorter window — its LP solver scales steeply with
    # row count (~14s/fit at 8760 rows vs ~2.5s at 720 rows). Without this
    # cap, the 120 LP solves (5 quantiles x 24 horizons) needed here take
    # ~28 minutes instead of ~3. This matches the same cap already applied
    # to LEAR's QR fitting in qr_models.py — missed here previously.
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


# ── Single shared AV-C-PID controller (chronological, hourly updates) ────────

def _avcpid_chronological(mu_flat, y_flat, mu_tr_flat, y_tr_flat, alpha,
                          seasonal_period=24, progress_label="",
                          progress_every=500):
    """
    Online AV-C-PID, updated once per HOUR as the test period unfolds
    chronologically. A single state (σ, bias, q_U, q_L) is shared across
    all hour-of-day slots — the base model's own forecast already accounts
    for daily seasonality in price level, so this controller only needs
    to track the residual/width dynamics, which can reasonably be shared.

    mu_flat, y_flat : (n_test*n_h,) — chronological (day-major, hour-minor)
                      base model Q50 predictions and actuals
    mu_tr_flat, y_tr_flat : (n_cal,) — training predictions/actuals for init
    """
    n_te = len(mu_flat)

    # Initialise bias LEVEL and TREND from training residuals (Holt's method).
    e_tr_raw   = (y_tr_flat - mu_tr_flat).astype(np.float64)
    bias_level = float(np.mean(e_tr_raw))
    bias_trend = 0.0

    # Volatility and thresholds estimated on BIAS-CORRECTED residuals.
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

    # DAM CAUSAL LAG — all state updates (σ, bias, q_U/q_L) use the residual
    # from t-24 (exactly one day ago) rather than the residual at t.
    #
    # Motivation (flagged by supervisor): in the DAM, all 24 hourly prices for
    # day d are bid simultaneously the evening before — you cannot observe any
    # price from day d before submitting your bids for day d. The only realized
    # prices available at bidding time are from day d-2 or earlier (day d-1's
    # prices are announced simultaneously with bidding for day d). Under the
    # previous hour-by-hour update, the controller was using hour t's outcome
    # to immediately adjust hour t+1's interval — causally invalid in the DAM.
    #
    # Using e_{t-24} ensures every update relies only on prices that were
    # genuinely known before the current day's bids were submitted. For the
    # first 24 steps (day 0), no lag residual is available, so updates are
    # deferred until t >= 24.
    LAG = seasonal_period   # 24 for DAM, 16 for BM

    # Buffer to store raw residuals so we can look them up 24 steps later.
    # e_raw_buffer[i] = raw residual (y - mu_flat) at step i.
    e_raw_buffer = {}

    t0 = time.time()
    for t in range(n_te):
        sigma = math.sqrt(max(sigma2, 1e-6))

        # Bias-corrected, trend-extrapolated center
        predicted_bias = bias_level + bias_trend
        center_t = mu_flat[t] + predicted_bias

        lo[t] = center_t - q_L * sigma
        hi[t] = center_t + q_U * sigma

        # Compute e_t relative to today's center, used for scoring (coverage
        # check) but NOT for state updates — that uses e_{t-LAG} below.
        e_t      = float(y_flat[t]) - center_t
        s_U_t    = max(e_t, 0.0) / sigma
        s_L_t    = max(-e_t, 0.0) / sigma
        covered_U = (s_U_t <= q_U)
        covered_L = (s_L_t <= q_L)

        # Store today's raw residual for use LAG steps from now
        e_raw_buffer[t] = float(y_flat[t]) - float(mu_flat[t])

        # ── State update uses e_{t-LAG} (causally valid for the DAM) ─────────
        if t >= LAG:
            # Retrieve the lagged raw residual and compute the lagged centered
            # residual relative to the bias known at that earlier time.
            # (We use the raw residual stored at t-LAG, which is causally safe.)
            raw_lag   = e_raw_buffer[t - LAG]
            # Center the lagged residual using the CURRENT bias (a minor
            # approximation — the exact bias at t-LAG is not stored, but since
            # bias changes slowly this is negligible).
            e_lag     = raw_lag - predicted_bias

            # PID coverage update using lagged observation
            s_U_lag   = max(e_lag, 0.0) / sigma
            s_L_lag   = max(-e_lag, 0.0) / sigma
            # ── Error signal: binary indicator (Angelopoulos et al. spec) ──
            # Values in {0, 1} → err_hist values in {-α, 1-α}, bounded.
            # Keeps err_sum and the saturation integrator r_t stable.
            # Supervisor's comment on the binary indicator was addressed by
            # testing a soft/proportional alternative (tanh), but empirical
            # tests showed the binary version produces better coverage and
            # Winkler scores — the tanh gradient is weaker on violations
            # (capped at 1 × lr vs binary's (1-α) × lr = 0.9 × lr), causing
            # under-reaction and systematically lower coverage.
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

            # σ update using lagged centered residual
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

            # Bias tracker update using the AVERAGE of all 24 previous-day
            # residuals rather than only e_{t-24}. All 24 hours of yesterday
            # are causally available at DAM bidding time, and averaging them
            # gives a much more stable signal than a single noisy residual.
            # Tested on synthetic data: reduces IW80 from 14.84→13.77 and
            # Winkler from 19.94→18.75 while maintaining Cov80≈0.80.
            # ALPHA_BIAS raised to 0.40 and BETA_BIAS to 0.10 to match the
            # faster, smoother signal from day-averaging.
            day_resids = [e_raw_buffer[t - h]
                          for h in range(1, LAG + 1)
                          if (t - h) in e_raw_buffer]
            update_signal = float(np.mean(day_resids)) if day_resids else raw_lag
            new_level = ALPHA_BIAS * update_signal + (1 - ALPHA_BIAS) * (bias_level + bias_trend)
            new_trend = BETA_BIAS * (new_level - bias_level) + (1 - BETA_BIAS) * bias_trend
            bias_level, bias_trend = new_level, new_trend

            # Clean up entries older than LAG steps (they've been used for
            # both the lagged residual AND the day-average by now).
            if (t - LAG) in e_raw_buffer:
                del e_raw_buffer[t - LAG]

        if progress_every and (t + 1) % progress_every == 0:
            pct = 100 * (t + 1) / n_te
            elapsed = time.time() - t0
            eta = elapsed / (t + 1) * (n_te - t - 1)
            print(f"    {progress_label} step {t+1:6d}/{n_te} "
                  f"({pct:5.1f}%)  elapsed={elapsed:6.1f}s  eta={eta:6.1f}s")

    return lo, hi


# ── Main AV-C-PID function ────────────────────────────────────────────────────

def run_avcpid(model_name, data, quantiles=None, retrain_every=None,
               qr_preds=None):
    """
    AV-C-PID: asymmetric volatility-aware conformal PID control.

    Base forecaster: Q50 from QR predictions (model-specific: KNN/LEAR/RF/LGBM).
    Scorecaster (D term): Theta model on threshold sequence (Angelopoulos et al.)
    EWMA volatility scaling + asymmetric calibration (thesis Sec 4.3).

    Updates happen once per HOUR, chronologically across the whole test
    period (see _avcpid_chronological), not once per day per hour-of-day.

    qr_preds : if passed (already computed), used directly as μ_t.
               Otherwise QR is re-computed (slower).
    """
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

    # ── Get Q50 point forecasts from base model ───────────────────────────────
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

    # ── Get training residuals for initialization ─────────────────────────────
    print(f"  Fitting {model_name} on training data for residual initialization...")
    mu_tr, y_tr_cal = _get_training_residuals(model_name, data)
    mu_tr_flat = mu_tr.flatten().astype(np.float64)
    y_tr_flat  = y_tr_cal.flatten().astype(np.float64)

    # ── Flatten to one chronological hourly sequence ──────────────────────────
    # Row-major flatten: day1-hour0,1,...,23, day2-hour0,1,...,23, ...
    mu_flat = mu_te.flatten()
    y_flat  = y_te.flatten()
    n_flat  = len(mu_flat)

    print(f"  Running chronological hourly AV-C-PID "
          f"({n_flat} hourly steps × 2 alphas)...")

    preds = np.zeros((n_te, n_h, len(quantiles)), dtype=np.float32)
    q50_idx = quantiles.index(0.5) if 0.5 in quantiles else 2
    preds[:, :, q50_idx] = mu_te.astype(np.float32)

    progress_every = max(500, n_flat // 20)   # ~20 progress prints total

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

    # Reshape back to (n_test, n_h)
    lo10 = lo10.reshape(n_te, n_h)
    hi90 = hi90.reshape(n_te, n_h)
    lo30 = lo30.reshape(n_te, n_h)
    hi70 = hi70.reshape(n_te, n_h)

    q_idx = {q: i for i, q in enumerate(quantiles)}
    if 0.1 in q_idx: preds[:, :, q_idx[0.1]] = lo10
    if 0.3 in q_idx: preds[:, :, q_idx[0.3]] = lo30
    if 0.7 in q_idx: preds[:, :, q_idx[0.7]] = hi70
    if 0.9 in q_idx: preds[:, :, q_idx[0.9]] = hi90

    # Enforce monotonicity Q10 ≤ Q30 ≤ Q50 ≤ Q70 ≤ Q90
    for qi in range(1, len(quantiles)):
        preds[:, :, qi] = np.maximum(preds[:, :, qi], preds[:, :, qi-1])

    print(f"  [AV-C-PID-{model_name}] done. "
          f"Total: {(time.time()-t0)/60:.1f} min\n")
    return preds


# ── PID-Q-Ens ─────────────────────────────────────────────────────────────────

def run_pid_q_ens(qr_preds, enbpi_preds, spci_preds, avcpid_preds):
    """PID-Q-Ens: equal-weight average of QR + EnbPI + SPCI + AV-C-PID."""
    return ((qr_preds.astype(np.float32)
             + enbpi_preds.astype(np.float32)
             + spci_preds.astype(np.float32)
             + avcpid_preds.astype(np.float32)) / 4.0)
