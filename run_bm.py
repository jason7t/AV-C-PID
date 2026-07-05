"""run_bm.py  —  Full BM pipeline."""
import os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

from config      import (RESULTS_DIR, MODEL_NAMES, METHOD_NAMES,
                          QUANTILES, BM_RETRAIN_EVERY)
from data_loader import load_bm
from qr_models   import run_qr
from cp_methods  import run_enbpi, run_spci, run_scp
from qra_methods import run_qra_r, run_qra_cp, run_q_ens
from avcpid    import run_avcpid, run_pid_q_ens
from metrics     import (compute_all_metrics, metrics_to_df,
                          print_aps_table, save_predictions_csv)
from trading_strategies import run_trading

METHOD_FN = {'QR': run_qr, 'EnbPI': run_enbpi, 'SPCI': run_spci,
             'SCP': run_scp, 'QRA-R': run_qra_r, 'QRA-CP': run_qra_cp,
             'AV-C-PID': run_avcpid}


def main():
    t0 = time.time()
    print("=" * 65)
    print(f"  EPF REPLICATION — BM")
    print(f"  Models : {MODEL_NAMES}")
    print(f"  Methods: {METHOD_NAMES}")
    print("=" * 65)

    data  = load_bm();  print()
    y_te  = data['y_test'];  cols = data['target_col_names']
    all_preds = {};  all_res = {};  all_trade = {}

    def _save(p, model, method):
        save_predictions_csv(p, y_te, cols, BM_RETRAIN_EVERY,
                             model, method, 'BM', RESULTS_DIR)

    for model in MODEL_NAMES:
        print(f"\n{'─'*65}\n  BASE MODEL: {model}\n{'─'*65}")
        all_preds[model] = {}

        for method in METHOD_NAMES:
            if method in ('Q-Ens', 'PID-Q-Ens'): continue
            t1 = time.time()
            # Pass pre-computed component predictions to avoid recomputation
            kwargs = {}
            if method == 'QRA-CP':
                kwargs = {'scp_preds':   all_preds[model].get('SCP'),
                          'enbpi_preds': all_preds[model].get('EnbPI'),
                          'spci_preds':  all_preds[model].get('SPCI')}
            elif method == 'AV-C-PID':
                kwargs = {'qr_preds': all_preds[model].get('QR')}
            p  = METHOD_FN[method](model, data, **kwargs)
            all_preds[model][method] = p
            _save(p, model, method)
            print(f"  → {method}-{model}: {time.time()-t1:.0f}s\n")

        if 'Q-Ens' in METHOD_NAMES:
            needed = {'QR', 'EnbPI', 'SPCI'}
            if needed.issubset(all_preds[model]):
                qe = run_q_ens(all_preds[model]['QR'],
                                all_preds[model]['EnbPI'],
                                all_preds[model]['SPCI'])
                all_preds[model]['Q-Ens'] = qe;  _save(qe, model, 'Q-Ens')
                print(f"  → Q-Ens-{model}: done\n")

        if 'PID-Q-Ens' in METHOD_NAMES:
            needed = {'QR', 'EnbPI', 'SPCI', 'AV-C-PID'}
            if needed.issubset(all_preds[model]):
                pqe = run_pid_q_ens(all_preds[model]['QR'],
                                     all_preds[model]['EnbPI'],
                                     all_preds[model]['SPCI'],
                                     all_preds[model]['AV-C-PID'])
                all_preds[model]['PID-Q-Ens'] = pqe;  _save(pqe, model, 'PID-Q-Ens')
                print(f"  → PID-Q-Ens-{model}: done\n")

    print(f"\n{'='*65}\n  METRICS\n{'='*65}")
    for model in MODEL_NAMES:
        all_res[model] = {};  all_trade[model] = {}
        for method in METHOD_NAMES:
            if method not in all_preds.get(model, {}): continue
            all_res[model][method]   = compute_all_metrics(y_te, all_preds[model][method], BM_RETRAIN_EVERY)
            all_trade[model][method] = run_trading(all_preds[model][method],
                                                    y_te, cols, BM_RETRAIN_EVERY, 'BM')

    print_aps_table(all_res)
    metrics_to_df(all_res).to_csv(os.path.join(RESULTS_DIR, 'BM_metrics.csv'))
    rows = [{'Model':m,'Method':mt,**all_trade[m][mt]}
            for m in MODEL_NAMES for mt in METHOD_NAMES
            if mt in all_trade.get(m,{})]
    pd.DataFrame(rows).set_index(['Model','Method']).to_csv(
        os.path.join(RESULTS_DIR, 'BM_trading.csv'))

    print(f"\n  Total: {(time.time()-t0)/60:.1f} min  |  Results in ./{RESULTS_DIR}/")


if __name__ == '__main__':
    main()
