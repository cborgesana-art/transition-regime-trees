"""Full-data three-state transition-regime tree (honest, 1-SE, K=5 inner folds, depth 4, 300 firms per leaf)."""
import os
import numpy as np, pandas as pd, sys, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trtrees.multistate import MultiStateTree
from sabi3_design import design
D = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/trans3.pkl'); ZC = [c for c in D.columns if c.startswith('z_')]
X1, X2 = design(D); y = D.y.values; g = D.firm.values; Z = D[ZC].values.astype(float); t0 = time.time()
t = MultiStateTree(max_depth=4, min_units_leaf=300, se_rule=1, n_folds=5, random_state=0, max_iter=8).fit(Z, X1, X2, y, g)
print(t.n_leaves(), [(d, ZC[v], round(th, 3)) for d, v, th in t.splits()], round(time.time() - t0), flush=True)
def strip(n):
    n = {k: v for k, v in n.items() if k != 'units'}
    for s in ('left', 'right'):
        if s in n: n[s] = strip(n[s])
    return n
t.tree_ = strip(t.tree_); pickle.dump(t, open(os.environ.get('OUT_DIR', 'out') + '/tree_full.pkl', 'wb'))
