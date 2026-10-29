"""Five-fold comparison over firms (same folds as the two-state analysis): global multinomial, honest tree, BIC, AIC,
LightGBM multiclass (unweighted, the same transition likelihood as the trees). Predictions of the three outcome probabilities for held-out firms."""
import numpy as np, pandas as pd, sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trtrees.multistate import MultiStateTree
from sabi3_design import design, XC
import lightgbm as lgb
D = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/trans3.pkl'); ZC = [c for c in D.columns if c.startswith('z_')]
X1, X2 = design(D); y = D.y.values.astype(int); g = D.firm.values; Z = D[ZC].values.astype(float); s = D.s_prev.values
firms = np.unique(g); rng = np.random.default_rng(2026); fold_of = dict(zip(firms, rng.permutation(len(firms)) % 5 + 1)); fo = np.array([fold_of[i] for i in g])
pd.DataFrame({'firm': firms, 'fold': [fold_of[f] for f in firms]}).to_csv(os.environ.get('OUT_DIR', 'out') + '/folds.csv', index=False)
which = sys.argv[1].split(',') if len(sys.argv) > 1 else ['global', 'honest', 'bic', 'aic', 'lgb']
F = np.column_stack([Z, D[XC].values, s])
import os as _os
for k in [int(x) for x in _os.environ.get('FOLDS', '1,2,3,4,5').split(',')]:
    tr = fo != k; te = fo == k
    for nm in which:
        out = os.environ.get('OUT_DIR', 'out') + f'/cv_{nm}_{k}.csv'
        if os.path.exists(out): continue
        t0 = time.time(); leaves = np.nan
        if nm == 'lgb':
            w = 1.0 / pd.Series(g[tr]).map(pd.Series(g[tr]).value_counts()).values
            m = lgb.LGBMClassifier(objective='multiclass', n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1)
            m.fit(F[tr], y[tr]); P = m.predict_proba(F[te])
        else:
            kw = dict(global_=dict(max_depth=0), honest=dict(criterion='honest', max_depth=4, se_rule=1, n_folds=5, random_state=k),
                      bic=dict(criterion='bic', max_depth=4), aic=dict(criterion='aic', max_depth=4))['global_' if nm == 'global' else nm]
            t = MultiStateTree(min_units_leaf=300, max_iter=8, **kw).fit(Z[tr], X1[tr], X2[tr], y[tr], g[tr]); P = t.predict_proba(Z[te], X1[te], X2[te]); leaves = t.n_leaves()
        pd.DataFrame(dict(firm=g[te], year=D.year.values[te], s_prev=s[te], y=y[te], p0=P[:, 0], p1=P[:, 1], p2=P[:, 2], leaves=leaves)).to_csv(out, index=False)
        print(k, nm, leaves, round(time.time() - t0), flush=True)
