"""resume_dam.py  —  Resume BM from any checkpoint."""
import os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

from config      import (RESULTS_DIR, MODEL_NAMES, METHOD_NAMES,
                          QUANTILES, QUANTILE_LABELS, BM_RETRAIN_EVERY)
from data_loader import load_bm
from qr_models   import run_qr
from cp_methods  import run_enbpi, run_spci, run_scp
from qra_methods import run_qra_r, run_qra_cp, run_q_ens
from avcpid    import run_avcpid, run_pid_q_ens
from metrics     import (compute_all_metrics, metrics_to_df,
                          print_aps_table, save_predictions_csv)
from trading_strategies import run_trading

MARKET = "BM"
METHOD_FN = {'QR':run_qr,'EnbPI':run_enbpi,'SPCI':run_spci,
             'SCP':run_scp,'QRA-R':run_qra_r,'QRA-CP':run_qra_cp,
             'AV-C-PID':run_avcpid}


def _load_csv(model, method, n_test, n_h, cols):
    fpath = os.path.join(RESULTS_DIR, f'{model.lower()}_{method}_{MARKET}.csv')
    if not os.path.exists(fpath): return None
    df = pd.read_csv(fpath)
    p  = np.zeros((len(df), n_h, len(QUANTILES)), dtype=np.float32)
    for qi, lbl in enumerate(QUANTILE_LABELS):
        for hi, col in enumerate(cols[:n_h]):
            key = f'{col}_Forecast_{lbl}'
            if key in df.columns:
                p[:, hi, qi] = df[key].values
    return np.repeat(p, BM_RETRAIN_EVERY, axis=0)[:n_test]


def main():
    t0   = time.time()
    data = load_bm(); print()
    y_te = data['y_test']; cols = data['target_col_names']
    n_te, n_h = len(y_te), data['n_horizons']
    all_preds = {m: {} for m in MODEL_NAMES}

    print("Loading completed predictions …")
    for model in MODEL_NAMES:
        for method in METHOD_NAMES:
            p = _load_csv(model, method, n_te, n_h, cols)
            if p is not None:
                all_preds[model][method] = p
                print(f"  Loaded  {model:5s} {method:8s}  {p.shape}")
    print()

    def _save(p, model, method):
        save_predictions_csv(p, y_te, cols, BM_RETRAIN_EVERY,
                             model, method, MARKET, RESULTS_DIR)

    def _run_or_skip(model, method):
        if method in all_preds[model]:
            print(f"  {method}-{model}: already done — skipping")
            return all_preds[model][method]
        kwargs = {}
        if method == 'QRA-CP':
            kwargs = {'scp_preds':   all_preds[model].get('SCP'),
                      'enbpi_preds': all_preds[model].get('EnbPI'),
                      'spci_preds':  all_preds[model].get('SPCI')}
        elif method == 'AV-C-PID':
            kwargs = {'qr_preds': all_preds[model].get('QR')}
        p = METHOD_FN[method](model, data, **kwargs)
        all_preds[model][method] = p;  _save(p, model, method)
        return p

    for model in MODEL_NAMES:
        todo = [m for m in METHOD_NAMES if m not in all_preds[model] and m != 'Q-Ens']
        if not todo and ('Q-Ens' not in METHOD_NAMES or 'Q-Ens' in all_preds[model]):
            print(f"  {model}: all done"); continue
        print(f"\n{'─'*65}\n  {model}  todo={todo}\n{'─'*65}")
        for method in METHOD_NAMES:
            if method in ('Q-Ens', 'PID-Q-Ens'): continue
            _run_or_skip(model, method)
        if 'Q-Ens' in METHOD_NAMES and 'Q-Ens' not in all_preds[model]:
            needed = {'QR','EnbPI','SPCI'}
            if needed.issubset(all_preds[model]):
                qe = run_q_ens(all_preds[model]['QR'],
                                all_preds[model]['EnbPI'],
                                all_preds[model]['SPCI'])
                all_preds[model]['Q-Ens'] = qe;  _save(qe, model, 'Q-Ens')
                print(f"  → Q-Ens-{model}: done\n")
        if 'PID-Q-Ens' in METHOD_NAMES and 'PID-Q-Ens' not in all_preds[model]:
            needed2 = {'QR','EnbPI','SPCI','AV-C-PID'}
            if needed2.issubset(all_preds[model]):
                pqe = run_pid_q_ens(all_preds[model]['QR'],all_preds[model]['EnbPI'],
                                     all_preds[model]['SPCI'],all_preds[model]['AV-C-PID'])
                all_preds[model]['PID-Q-Ens'] = pqe;  _save(pqe, model, 'PID-Q-Ens')
                print(f"  → PID-Q-Ens-{model}: done\n")

    all_res = {};  all_trade = {}
    for model in MODEL_NAMES:
        all_res[model] = {};  all_trade[model] = {}
        for method in METHOD_NAMES:
            if method not in all_preds.get(model, {}): continue
            all_res[model][method]   = compute_all_metrics(y_te, all_preds[model][method], BM_RETRAIN_EVERY)
            all_trade[model][method] = run_trading(all_preds[model][method],
                                                    y_te, cols, BM_RETRAIN_EVERY, MARKET)
    print_aps_table(all_res)
    metrics_to_df(all_res).to_csv(os.path.join(RESULTS_DIR, f'{MARKET}_metrics.csv'))
    rows = [{'Model':m,'Method':mt,**all_trade[m][mt]}
            for m in MODEL_NAMES for mt in METHOD_NAMES
            if mt in all_trade.get(m,{})]
    pd.DataFrame(rows).set_index(['Model','Method']).to_csv(
        os.path.join(RESULTS_DIR, f'{MARKET}_trading.csv'))
    print(f"\n  Session: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
