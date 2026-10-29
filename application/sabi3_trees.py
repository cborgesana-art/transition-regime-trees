"""Refit the honest tree of each outer fold (same settings and seeds as sabi3_cv.py) and save it, for the decomposition."""
import numpy as np, pandas as pd, sys, os, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trtrees.multistate import MultiStateTree
from sabi3_design import design
D = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/trans3.pkl'); ZC = [c for c in D.columns if c.startswith('z_')]
X1, X2 = design(D); y = D.y.values.astype(int); g = D.firm.values; Z = D[ZC].values.astype(float)
fo = D.firm.map(pd.read_csv(os.environ.get('OUT_DIR', 'out') + '/folds.csv').set_index('firm').fold).values
def strip(n):
    n = {k: v for k, v in n.items() if k != 'units'}
    for s in ('left', 'right'):
        if s in n: n[s] = strip(n[s])
    return n
for k in [int(x) for x in sys.argv[1].split(',')]:
    out = os.environ.get('OUT_DIR', 'out') + f'/tree_honest_{k}.pkl'
    if os.path.exists(out): continue
    tr = fo != k; t0 = time.time()
    t = MultiStateTree(criterion='honest', max_depth=4, se_rule=1, n_folds=5, random_state=k, min_units_leaf=300, max_iter=8).fit(Z[tr], X1[tr], X2[tr], y[tr], g[tr])
    t.tree_ = strip(t.tree_); pickle.dump(t, open(out, 'wb')); print(k, t.n_leaves(), round(time.time() - t0), flush=True)
