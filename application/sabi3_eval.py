import numpy as np, pandas as pd, glob, os
from sklearn.metrics import roc_auc_score
M = {}
for nm in ['global', 'honest', 'bic', 'aic', 'mob', 'cox', 'lgb', 'gspline', 'tspline', 'lgb_noz', 'honest_off', 'bic_off', 'aic_off']:
    fs = sorted(glob.glob(os.environ.get('OUT_DIR', 'out') + f'/cv_{nm}_[0-9].csv'))
    if len(fs) == 5: M[nm] = pd.concat([pd.read_csv(f) for f in fs]).sort_values(['firm', 'year']).reset_index(drop=True)
print('methods', list(M))
rows = []; F = {}
for nm, d in M.items():
    P = d[['p0', 'p1', 'p2']].values.clip(1e-12); P = P / P.sum(1, keepdims=True); y = d.y.values.astype(int); s = d.s_prev.values
    ll = -np.log(P[np.arange(len(y)), y]); F[nm] = pd.Series(ll).groupby(d.firm.values).sum()
    h, st = s == 0, s == 1
    rows.append(dict(method=nm, traj_loss=F[nm].mean(), logloss_obs=ll.mean(), auc_entry=roc_auc_score(y[h] == 1, P[h, 1]), auc_exit_h=roc_auc_score(y[h] == 2, P[h, 2]),
                     auc_recovery=roc_auc_score(y[st] == 0, P[st, 0]), auc_exit_s=roc_auc_score(y[st] == 2, P[st, 2]),
                     cal_exit=P[:, 2].mean() / (y == 2).mean(), leaves=d.leaves.mean()))
T = pd.DataFrame(rows).sort_values('traj_loss'); print(T.round(4).to_string()); T.to_csv(os.environ.get('OUT_DIR', 'out') + '/compare3.csv', index=False)
import os
REF = os.environ.get('REF', 'honest')
if REF in F:
    rng = np.random.default_rng(1); out = []
    for nm in F:
        if nm == REF: continue
        dd = (F[REF] - F[nm]).dropna().values; n = len(dd); bs = [dd[rng.integers(0, n, n)].mean() for _ in range(2000)]
        out.append(dict(comparator=nm, mean_diff=dd.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), share_better=(dd < 0).mean()))
    O = pd.DataFrame(out); print(O.round(5).to_string()); O.to_csv(os.environ.get('OUT_DIR', 'out') + f'/diff_ci3_{REF}.csv', index=False)
