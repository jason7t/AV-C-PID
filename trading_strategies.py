"""
trading_strategies.py  —  All 4 trading strategies from O'Connor et al.

TS1 (Single_Trade)    : 1 buy-sell pair per day, dual-pair selection
TS2 (Multi_Trade)     : 3 trades per day (T1 + T2 before + T3 after)
TS3 (High_Frequency)  : recursive bottleneck-controlled strategy
TS4 (Dual_Strategy)   : combined DAM + BM strategy

Paper params: eff_1=0.8, eff_2=0.98, skip 152 DAM / 456 BM rows.
"""
import numpy as np
import pandas as pd
from config import (EFF_1, EFF_2, BATTERY_CAP, RAMP_RATE, MIN_SOC, INITIAL_SOC,
                    TRADING_SKIP_DAM, TRADING_SKIP_BM,
                    QUANTILES, QUANTILE_LABELS, RESULTS_DIR)


# ── Data preparation helpers ──────────────────────────────────────────────────

def preds_to_trading_df(preds, y_test, target_col_names, retrain_every, market):
    """
    Convert prediction array to paper's trading format.
    Returns (Y_r, Q_10, Q_30, Q_50, Q_70, Q_90) DataFrames with 'level_0' and 'Price'.
    """
    n_te, n_h, n_q = preds.shape
    skip = TRADING_SKIP_DAM if market == 'DAM' else TRADING_SKIP_BM

    batch_starts = list(range(0, n_te, retrain_every))

    # Build wide DataFrame
    rows_actual = {}
    for hi, col in enumerate(target_col_names):
        rows_actual[col] = [float(y_test[bs, hi]) for bs in batch_starts]

    rows_q = {}
    for qi, lbl in enumerate(QUANTILE_LABELS):
        for hi, col in enumerate(target_col_names):
            rows_q[f'{col}_Forecast_{lbl}'] = [
                float(preds[bs, hi, qi]) for bs in batch_starts]

    wide = pd.DataFrame({**rows_actual, **rows_q})

    # Skip warm-up rows
    wide = wide.iloc[skip:].reset_index(drop=True)

    def _stack_q(lbl):
        cols = [f'{c}_Forecast_{lbl}' for c in target_col_names]
        df   = wide[cols].dropna().stack().reset_index()
        df.columns = ['level_0', 'level_1', 'value']
        df['Price'] = df['value']
        return df

    def _stack_actual():
        df   = wide[target_col_names].dropna().stack().reset_index()
        df.columns = ['level_0', 'level_1', 'value']
        df['Price'] = df['value']
        return df

    return (_stack_actual(),
            _stack_q(10), _stack_q(30), _stack_q(50), _stack_q(70), _stack_q(90))


# ── TS1: Single Trade ─────────────────────────────────────────────────────────

def _single_trade_period(df, Q_A, Q_B, eff_1, eff_2, period_col='level_0'):
    """Run TS1 for all periods. Returns DataFrame with profit per trade."""
    prices = []
    for period in df[period_col].unique():
        cur_df = df[df[period_col] == period]
        cur_A  = Q_A[Q_A[period_col] == period]
        cur_B  = Q_B[Q_B[period_col] == period]
        if cur_A.empty or cur_B.empty:
            continue

        max_idx = cur_A['Price'].idxmax()
        min_idx = cur_B['Price'].idxmin()

        before_max = Q_B[(Q_B[period_col] == period) & (Q_B.index < max_idx)]
        after_min  = Q_A[(Q_A[period_col] == period) & (Q_A.index > min_idx)]

        min_idx1 = before_max['Price'].idxmin() if not before_max.empty else None
        max_idx1 = after_min['Price'].idxmax()  if not after_min.empty  else None

        chosen_max = chosen_min = None

        if all(x is not None for x in [max_idx, min_idx1, max_idx1]):
            d1 = cur_A.loc[max_idx, 'Price']  - cur_B.loc[min_idx1, 'Price']
            d2 = cur_A.loc[max_idx1, 'Price'] - cur_B.loc[min_idx,  'Price']
            if d1 > d2:
                chosen_max, chosen_min = max_idx, min_idx1
            else:
                chosen_max, chosen_min = max_idx1, min_idx
        elif max_idx is not None and min_idx1 is not None:
            chosen_max, chosen_min = max_idx, min_idx1
        elif max_idx1 is not None and min_idx is not None:
            chosen_max, chosen_min = max_idx1, min_idx

        if chosen_max is None or chosen_min is None:
            continue
        if chosen_max not in cur_df.index or chosen_min not in cur_df.index:
            continue

        profit = (cur_df.loc[chosen_max, 'Price'] * eff_1 -
                  cur_df.loc[chosen_min, 'Price'] / eff_2)
        prices.append((chosen_min, cur_df.loc[chosen_min, 'Price'],
                        chosen_max, cur_df.loc[chosen_max, 'Price'], profit))

    return pd.DataFrame(prices, columns=['minIdx', 'minPrice',
                                          'maxIdx', 'maxPrice', 'profit'])


def run_ts1(preds, y_test, target_col_names, retrain_every, market):
    """TS1: Single trade with dual-pair selection."""
    Y_r, Q10, Q30, Q50, Q70, Q90 = preds_to_trading_df(
        preds, y_test, target_col_names, retrain_every, market)

    pairs = {
        '50-50': (Q50, Q50), '10-30': (Q10, Q30), '30-50': (Q30, Q50),
        '50-70': (Q50, Q70), '70-90': (Q70, Q90), '30-70': (Q30, Q70),
        '10-90': (Q10, Q90), 'PF':    (Y_r, Y_r),
    }
    results = {}
    for name, (A, B) in pairs.items():
        e1 = EFF_1 if name != 'PF' else 1.0
        e2 = EFF_2 if name != 'PF' else 1.0
        r  = _single_trade_period(Y_r, A, B, e1, e2)
        results[name] = float(r['profit'].sum()) if not r.empty else 0.0
    return results


# ── TS2: Multi-Trade ──────────────────────────────────────────────────────────

def _multi_trade_period(df, Q_A, Q_B, eff_1, eff_2, period_col='level_0'):
    """TS2: Up to 3 trades per period (T1 + T2 before + T3 after T1)."""
    prices = []
    for period in df[period_col].unique():
        cur_df = df[df[period_col] == period]
        cur_A  = Q_A[Q_A[period_col] == period]
        cur_B  = Q_B[Q_B[period_col] == period]
        if cur_A.empty or cur_B.empty:
            continue

        max_idx  = cur_A['Price'].idxmax()
        min_idx  = cur_B['Price'].idxmin()
        bef_max  = Q_B[(Q_B[period_col] == period) & (Q_B.index < max_idx)]
        aft_min  = Q_A[(Q_A[period_col] == period) & (Q_A.index > min_idx)]
        min_idx1 = bef_max['Price'].idxmin() if not bef_max.empty else None
        max_idx1 = aft_min['Price'].idxmax() if not aft_min.empty else None

        # T1: main trade
        T1_max = T1_min = None
        if all(x is not None for x in [max_idx, min_idx1, max_idx1]):
            d1 = cur_A.loc[max_idx,'Price']  - cur_B.loc[min_idx1,'Price']
            d2 = cur_A.loc[max_idx1,'Price'] - cur_B.loc[min_idx,'Price']
            if d1 > d2:
                T1_max, T1_min = max_idx, min_idx1
            else:
                T1_max, T1_min = max_idx1, min_idx
        elif max_idx is not None and min_idx1 is not None:
            T1_max, T1_min = max_idx, min_idx1
        elif max_idx1 is not None and min_idx is not None:
            T1_max, T1_min = max_idx1, min_idx
        if T1_max is None or T1_min is None:
            continue
        if T1_max in cur_df.index and T1_min in cur_df.index:
            profit = (cur_df.loc[T1_max,'Price'] * eff_1 -
                      cur_df.loc[T1_min,'Price'] / eff_2)
            prices.append((T1_min, cur_df.loc[T1_min,'Price'],
                            T1_max, cur_df.loc[T1_max,'Price'], profit))

        # T2: trade in prices BEFORE T1_min
        cA_bef = Q_A[(Q_A[period_col]==period) & (Q_A.index < T1_min)]
        cB_bef = Q_B[(Q_B[period_col]==period) & (Q_B.index < T1_min)]
        if not cA_bef.empty and not cB_bef.empty:
            T2_max_raw = cA_bef['Price'].idxmax()
            T2_min_raw = cB_bef['Price'].idxmin()
            bef2 = Q_B[(Q_B[period_col]==period) & (Q_B.index < T2_max_raw)]
            aft2 = Q_A[(Q_A[period_col]==period) & (Q_A.index > T2_min_raw)
                        & (Q_A.index < T1_min)]
            T2_min2 = bef2['Price'].idxmin() if not bef2.empty else None
            T2_max2 = aft2['Price'].idxmax() if not aft2.empty else None
            T2_max = T2_min = None
            if all(x is not None for x in [T2_max_raw, T2_min2, T2_max2]):
                d1 = cur_A.loc[T2_max_raw,'Price'] - cur_B.loc[T2_min2,'Price']
                d2 = cur_A.loc[T2_max2,'Price']    - cur_B.loc[T2_min_raw,'Price']
                T2_max, T2_min = (T2_max_raw, T2_min2) if d1>d2 else (T2_max2, T2_min_raw)
            elif T2_max_raw is not None and T2_min2 is not None:
                T2_max, T2_min = T2_max_raw, T2_min2
            elif T2_max2 is not None and T2_min_raw is not None:
                T2_max, T2_min = T2_max2, T2_min_raw
            if T2_max is not None and T2_min is not None:
                if T2_max in cur_df.index and T2_min in cur_df.index:
                    profit = (cur_df.loc[T2_max,'Price'] * eff_1 -
                              cur_df.loc[T2_min,'Price'] / eff_2)
                    prices.append((T2_min, cur_df.loc[T2_min,'Price'],
                                    T2_max, cur_df.loc[T2_max,'Price'], profit))

        # T3: trade in prices AFTER T1_max
        cA_aft = Q_A[(Q_A[period_col]==period) & (Q_A.index > T1_max)]
        cB_aft = Q_B[(Q_B[period_col]==period) & (Q_B.index > T1_max)]
        if not cA_aft.empty and not cB_aft.empty:
            T3_max_raw = cA_aft['Price'].idxmax()
            T3_min_raw = cB_aft['Price'].idxmin()
            bef3 = Q_B[(Q_B[period_col]==period) & (Q_B.index < T3_max_raw)
                        & (Q_B.index > T1_max)]
            aft3 = Q_A[(Q_A[period_col]==period) & (Q_A.index > T3_min_raw)]
            T3_min2 = bef3['Price'].idxmin() if not bef3.empty else None
            T3_max2 = aft3['Price'].idxmax() if not aft3.empty else None
            T3_max = T3_min = None
            if all(x is not None for x in [T3_max_raw, T3_min2, T3_max2]):
                d1 = cur_A.loc[T3_max_raw,'Price'] - cur_B.loc[T3_min2,'Price']
                d2 = cur_A.loc[T3_max2,'Price']    - cur_B.loc[T3_min_raw,'Price']
                T3_max, T3_min = (T3_max_raw, T3_min2) if d1>d2 else (T3_max2, T3_min_raw)
            elif T3_max_raw is not None and T3_min2 is not None:
                T3_max, T3_min = T3_max_raw, T3_min2
            elif T3_max2 is not None and T3_min_raw is not None:
                T3_max, T3_min = T3_max2, T3_min_raw
            if T3_max is not None and T3_min is not None:
                if T3_max in cur_df.index and T3_min in cur_df.index:
                    profit = (cur_df.loc[T3_max,'Price'] * eff_1 -
                              cur_df.loc[T3_min,'Price'] / eff_2)
                    prices.append((T3_min, cur_df.loc[T3_min,'Price'],
                                    T3_max, cur_df.loc[T3_max,'Price'], profit))

    return pd.DataFrame(prices, columns=['minIdx','minPrice','maxIdx','maxPrice','profit'])


def run_ts2(preds, y_test, target_col_names, retrain_every, market):
    """TS2: Multi-trade (up to 3 trades per period)."""
    Y_r, Q10, Q30, Q50, Q70, Q90 = preds_to_trading_df(
        preds, y_test, target_col_names, retrain_every, market)
    pairs = {
        '50-50': (Q50,Q50), '10-30': (Q10,Q30), '30-50': (Q30,Q50),
        '50-70': (Q50,Q70), '70-90': (Q70,Q90), '30-70': (Q30,Q70),
        '10-90': (Q10,Q90), 'PF':    (Y_r,Y_r),
    }
    results = {}
    for name, (A, B) in pairs.items():
        e1 = EFF_1 if name != 'PF' else 1.0
        e2 = EFF_2 if name != 'PF' else 1.0
        r  = _multi_trade_period(Y_r, A, B, e1, e2)
        results[name] = float(r['profit'].sum()) if not r.empty else 0.0
    return results


# ── TS3: High-Frequency (bottleneck-controlled) ───────────────────────────────

def _process_prices(charge, cap, ramp, min_soc, e1, e2, prices, cur_df, min_i, max_i):
    # Matches O'Connor's process_prices_DAM/BM exactly: if/elif on min_i vs
    # max_i, with NO trade executed when min_i == max_i (charge unchanged,
    # nothing appended). Our previous if/else collapsed '==' into the '>'
    # branch, incorrectly executing a trade in that edge case.
    if min_i < max_i:
        b1 = min(cap - charge, ramp);  charge += b1
        b2 = min(charge - min_soc, ramp); charge -= b2
        profit = cur_df.loc[max_i,'Price']*b2*e1 - cur_df.loc[min_i,'Price']*b1/e2
        prices.append((min_i, cur_df.loc[min_i,'Price'],
                        max_i, cur_df.loc[max_i,'Price'], profit, charge))
    elif min_i > max_i:
        b2 = min(charge - min_soc, ramp); charge -= b2
        b1 = min(cap - charge, ramp);  charge += b1
        profit = cur_df.loc[max_i,'Price']*b2*e1 - cur_df.loc[min_i,'Price']*b1/e2
        prices.append((min_i, cur_df.loc[min_i,'Price'],
                        max_i, cur_df.loc[max_i,'Price'], profit, charge))
    # min_i == max_i: no trade, charge unchanged (matches O'Connor exactly)
    return charge


def _recursive_hf(charge, cap, ramp, min_soc, e1, e2, prices,
                   cur_df, rA, rB, cur_A, cur_B, period):
    if len(rA) <= 1: return charge
    max_i = rA['Price'].idxmax()
    min_i = rB['Price'].idxmin()
    if cur_B.loc[min_i,'Price'] < cur_A.loc[max_i,'Price']:
        charge = _process_prices(charge, cap, ramp, min_soc,
                                  e1, e2, prices, cur_df, min_i, max_i)
    s, l = min(min_i, max_i), max(min_i, max_i)
    rA = cur_A[(cur_A['level_0']==period) & (cur_A.index>s) & (cur_A.index<l)]
    rB = cur_B[(cur_B['level_0']==period) & (cur_B.index>s) & (cur_B.index<l)]
    return _recursive_hf(charge, cap, ramp, min_soc,
                          e1, e2, prices, cur_df, rA, rB, cur_A, cur_B, period)


def _hf_strategy(df, Q_A, Q_B, eff_1, eff_2,
                  cap=BATTERY_CAP, charge=INITIAL_SOC,
                  ramp=RAMP_RATE, min_soc=MIN_SOC):
    prices = []
    for period in df['level_0'].unique():
        cur_df = df[df['level_0']==period]
        cur_A  = Q_A[Q_A['level_0']==period]
        cur_B  = Q_B[Q_B['level_0']==period]
        if cur_A.empty or cur_B.empty: continue
        max_i = cur_A['Price'].idxmax()
        min_i = cur_B['Price'].idxmin()
        s, l  = min(min_i, max_i), max(min_i, max_i)
        bef_A = Q_A[(Q_A['level_0']==period) & (Q_A.index<s)]
        bef_B = Q_B[(Q_B['level_0']==period) & (Q_B.index<s)]
        bet_A = Q_A[(Q_A['level_0']==period) & (Q_A.index>s) & (Q_A.index<l)]
        bet_B = Q_B[(Q_B['level_0']==period) & (Q_B.index>s) & (Q_B.index<l)]
        aft_A = Q_A[(Q_A['level_0']==period) & (Q_A.index>l)]
        aft_B = Q_B[(Q_B['level_0']==period) & (Q_B.index>l)]
        if len(bef_A) > 1:
            mi3 = bef_B['Price'].idxmin()
            ma3 = bef_A['Price'].idxmax()
            if cur_B.loc[mi3,'Price'] < cur_A.loc[ma3,'Price']:
                charge = _process_prices(charge, cap, ramp, min_soc,
                                          eff_1, eff_2, prices, cur_df, mi3, ma3)
            else: continue
        if cur_B.loc[min_i,'Price'] < cur_A.loc[max_i,'Price']:
            charge = _process_prices(charge, cap, ramp, min_soc,
                                      eff_1, eff_2, prices, cur_df, min_i, max_i)
        else: continue
        if len(bet_A) > 1:
            charge = _recursive_hf(charge, cap, ramp, min_soc,
                                     eff_1, eff_2, prices, cur_df,
                                     bet_A, bet_B, cur_A, cur_B, period)
        else: continue
        if len(aft_A) > 1:
            ma1 = aft_A['Price'].idxmax()
            mi1 = aft_B['Price'].idxmin()
            if cur_B.loc[mi1,'Price'] < cur_A.loc[ma1,'Price']:
                charge = _process_prices(charge, cap, ramp, min_soc,
                                          eff_1, eff_2, prices, cur_df, mi1, ma1)
            else: continue
    cols = ['minIdx','minPrice','maxIdx','maxPrice','profit','charge']
    return pd.DataFrame(prices, columns=cols)


def run_ts3(preds, y_test, target_col_names, retrain_every, market):
    """TS3: High-frequency bottleneck-controlled strategy."""
    Y_r, Q10, Q30, Q50, Q70, Q90 = preds_to_trading_df(
        preds, y_test, target_col_names, retrain_every, market)
    pairs = {
        '50-50': (Q50,Q50), '10-30': (Q10,Q30), '30-50': (Q30,Q50),
        '50-70': (Q50,Q70), '70-90': (Q70,Q90), '30-70': (Q30,Q70),
        '10-90': (Q10,Q90), 'PF':    (Y_r,Y_r),
    }
    results = {}
    for name, (A, B) in pairs.items():
        e1 = EFF_1 if name != 'PF' else 0.8
        e2 = EFF_2 if name != 'PF' else 0.98
        r  = _hf_strategy(Y_r, A, B, e1, e2)
        results[name] = float(r['profit'].sum()) if not r.empty else 0.0
    return results


# ── Run all trading strategies ────────────────────────────────────────────────

def run_trading(preds, y_test, target_col_names, retrain_every, market):
    """
    Run only TS1 and TS2, matching O'Connor et al. (2025)'s Table 6.

    NOTE on naming: our codebase internally has THREE strategy functions
    (run_ts1, run_ts2, run_ts3), matching variable names used during
    development against O'Connor's GitHub. However, the PEPF paper
    (o2025conformal) only reports two strategies in its results table,
    which it calls "TS1" and "TS2" — and their TS2 is the BOTTLENECK-
    CONSTRAINED strategy (battery capacity/ramp/min-SOC), which in our
    code is the function run_ts3 (matching O'Connor's GitHub function
    name `electricity_strategy_*_HF`, also internally called "TS3" in
    their own code/plots — but NOT reported as a separate strategy in
    the main PEPF paper's Table 6).

    Our code's run_ts2 (multi-trade, no bottleneck constraints) is
    therefore NOT included here, since it doesn't correspond to either
    of the two strategies in O'Connor et al. (2025)'s reported results.

    Output key 'TS2_*' below = run_ts3() = the bottleneck-constrained
    strategy = O'Connor PEPF paper's "TS2".
    """
    ts1 = run_ts1(preds, y_test, target_col_names, retrain_every, market)
    ts2 = run_ts3(preds, y_test, target_col_names, retrain_every, market)
    return {f'TS1_{k}': v for k, v in ts1.items()} | \
           {f'TS2_{k}': v for k, v in ts2.items()}
