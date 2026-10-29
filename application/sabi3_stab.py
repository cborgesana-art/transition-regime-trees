"""Stability: the three-state tree regrown on random halves of the firms (min 150 firms per leaf)."""
import numpy as np, pandas as pd, sys, time, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trtrees.multistate import MultiStateTree
from sabi3_design import design
from sabi3_offset import SplineOffset
D = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/trans3.pkl'); ZC = [c for c in D.columns if c.startswith('z_')]
X1, X2 = design(D); y = D.y.values; g = D.firm.values; Z = D[ZC].values.astype(float); firms = np.unique(g)
F = os.environ.get('OUT_DIR', 'out') + '/stab3off' + ('' if len(sys.argv) < 3 else '_' + sys.argv[2]) + '.csv'; out = pd.read_csv(F).to_dict('records') if os.path.exists(F) else []
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for b in range(lo, int(sys.argv[1])):
    if b in [o['rep'] for o in out]: continue
    t0 = time.time(); rng = np.random.default_rng(900 + b); sel = np.isin(g, rng.choice(firms, len(firms) // 2, replace=False))
    rows = np.where(sel)[0]; O = SplineOffset().fit(D, rows).offset(D, rows)
    t = MultiStateTree(max_depth=4, min_units_leaf=150, se_rule=1, n_folds=5, random_state=b, max_iter=8).fit(Z[sel], X1[sel], X2[sel], y[sel], g[sel], offset=O)
    out.append(dict(rep=b, leaves=t.n_leaves(), splits=json.dumps([(int(d), ZC[v], float(th)) for d, v, th in t.splits()])))
    pd.DataFrame(out).to_csv(F, index=False); print(b, t.n_leaves(), [(d, ZC[v][2:], round(th, 3)) for d, v, th in t.splits()][:4], round(time.time() - t0), flush=True)
