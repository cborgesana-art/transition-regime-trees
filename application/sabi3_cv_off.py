"""Outer five-fold comparison for trees whose leaves model deviations from a global spline model (offset)."""
import numpy as np, pandas as pd, sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trtrees.multistate import MultiStateTree
from sabi3_design import design
from sabi3_offset import SplineOffset
D = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/trans3.pkl'); DROP = os.environ.get('DROP', ''); ZC = [c for c in D.columns if c.startswith('z_') and c not in DROP.split(',')]; TAG = os.environ.get('TAG', '')
X1, X2 = design(D); y = D.y.values.astype(int); g = D.firm.values; Z = D[ZC].values.astype(float); s = D.s_prev.values
fo = D.firm.map(pd.read_csv(os.environ.get('OUT_DIR', 'out') + '/folds.csv').set_index('firm').fold).values
which = sys.argv[1].split(','); folds = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [1, 2, 3, 4, 5]
def strip(n):
    n = {k: v for k, v in n.items() if k != 'units'}
    for c in ('left', 'right'):
        if c in n: n[c] = strip(n[c])
    return n
for k in folds:
    tr = np.where(fo != k)[0]; te = np.where(fo == k)[0]; so = SplineOffset().fit(D, tr); Otr = so.offset(D, tr); Ote = so.offset(D, te)
    for nm in which:
        out = os.environ.get('OUT_DIR', 'out') + f'/cv_{nm}{TAG}_{k}.csv'
        if os.path.exists(out): continue
        t0 = time.time()
        kw = dict(honest_off=dict(criterion='honest', se_rule=1, n_folds=5, random_state=k), bic_off=dict(criterion='bic'), aic_off=dict(criterion='aic'))[nm]
        t = MultiStateTree(max_depth=4, min_units_leaf=300, max_iter=8, **kw).fit(Z[tr], X1[tr], X2[tr], y[tr], g[tr], offset=Otr)
        P = t.predict_proba(Z[te], X1[te], X2[te], offset=Ote)
        pd.DataFrame(dict(firm=g[te], year=D.year.values[te], s_prev=s[te], y=y[te], p0=P[:, 0], p1=P[:, 1], p2=P[:, 2], leaves=t.n_leaves())).to_csv(out, index=False)
        if nm == 'honest_off': t.tree_ = strip(t.tree_); t.O_ = None; pickle.dump(t, open(os.environ.get('OUT_DIR', 'out') + f'/tree_honest_off{TAG}_{k}.pkl', 'wb'))
        print(k, nm, t.n_leaves(), round(time.time() - t0), flush=True)
