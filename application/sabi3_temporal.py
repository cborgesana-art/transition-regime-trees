"""Temporal validation: for each outer fold k, fit on the transitions up to 2018 of the firms in the other folds and
evaluate on the transitions from 2019 to 2024 of the firms in fold k."""
import numpy as np, pandas as pd, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trtrees.multistate import MultiStateTree
from sabi3_design import design, XC
from sabi3_offset import SplineOffset
import lightgbm as lgb
D = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/trans3.pkl'); ZC = [c for c in D.columns if c.startswith('z_')]
X1, X2 = design(D); y = D.y.values.astype(int); g = D.firm.values; Z = D[ZC].values.astype(float); s = D.s_prev.values; yr = D.year.values
fo = D.firm.map(pd.read_csv(os.environ.get('OUT_DIR', 'out') + '/folds.csv').set_index('firm').fold).values
which = sys.argv[1].split(','); folds = [int(x) for x in sys.argv[2].split(',')]
for k in folds:
    tr = np.where((fo != k) & (yr <= 2018))[0]; te = np.where((fo == k) & (yr >= 2019))[0]
    so = SplineOffset().fit(D, tr); Otr = so.offset(D, tr); Ote = so.offset(D, te)
    for nm in which:
        out = os.environ.get('OUT_DIR', 'out') + f'/tv_{nm}_{k}.csv'
        if os.path.exists(out): continue
        t0 = time.time(); leaves = np.nan
        if nm == 'gspline':
            E1, E2 = Ote[:, 0], Ote[:, 1]; m = np.maximum(0, np.maximum(E1, E2)); A = np.column_stack([np.exp(-m), np.exp(E1 - m), np.exp(E2 - m)]); P = A / A.sum(1, keepdims=True)
        elif nm == 'lgb':
            F = np.column_stack([Z, D[XC].values, s])
            mdl = lgb.LGBMClassifier(objective='multiclass', n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1)
            mdl.fit(F[tr], y[tr]); P = mdl.predict_proba(F[te])
        else:
            kw = dict(honest_off=dict(criterion='honest', se_rule=1, n_folds=5, random_state=k), bic_off=dict(criterion='bic'), aic_off=dict(criterion='aic'))[nm]
            t = MultiStateTree(max_depth=4, min_units_leaf=300, max_iter=8, **kw).fit(Z[tr], X1[tr], X2[tr], y[tr], g[tr], offset=Otr)
            P = t.predict_proba(Z[te], X1[te], X2[te], offset=Ote); leaves = t.n_leaves()
        pd.DataFrame(dict(firm=g[te], year=yr[te], s_prev=s[te], y=y[te], p0=P[:, 0], p1=P[:, 1], p2=P[:, 2], leaves=leaves)).to_csv(out, index=False)
        print(k, nm, leaves, round(time.time() - t0), flush=True)
