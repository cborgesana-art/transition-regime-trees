"""Stability of the top of the tree: depth-two trees (around a spline offset refitted on each half) on random halves of
the firms; split frequencies, thresholds, and the adjusted Rand index between the partition of all firms by the
subsample tree and by the first two levels of the full-data tree."""
import numpy as np, pandas as pd, sys, time, json, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trtrees.multistate import MultiStateTree
from sabi3_design import design
from sabi3_offset import SplineOffset
from sklearn.metrics import adjusted_rand_score
D = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/trans3.pkl'); ZC = [c for c in D.columns if c.startswith('z_')]
X1, X2 = design(D); y = D.y.values; g = D.firm.values; Z = D[ZC].values.astype(float); firms, first = np.unique(g, return_index=True); Zf = Z[first]
full = pickle.load(open(os.environ.get('OUT_DIR', 'out') + '/tree_full_off.pkl', 'rb'))
def path(node, z, depth):
    p = ''
    for _ in range(depth):
        if 'var' not in node: break
        if z[node['var']] <= node['thr']: node = node['left']; p += 'L'
        else: node = node['right']; p += 'R'
    return p
ref = np.array([path(full.tree_, z, 2) for z in Zf])
lo, hi = int(sys.argv[1]), int(sys.argv[2]); F = os.environ.get('OUT_DIR', 'out') + f'/stab2_{lo}.csv'
out = pd.read_csv(F).to_dict('records') if os.path.exists(F) else []
for b in range(lo, hi):
    if b in [o['rep'] for o in out]: continue
    t0 = time.time(); rng = np.random.default_rng(5000 + b); sel = np.isin(g, rng.choice(firms, len(firms) // 2, replace=False))
    rows = np.where(sel)[0]; O = SplineOffset().fit(D, rows).offset(D, rows)
    t = MultiStateTree(max_depth=2, min_units_leaf=150, se_rule=1, n_folds=5, random_state=b, max_iter=8).fit(Z[sel], X1[sel], X2[sel], y[sel], g[sel], offset=O)
    lab = np.array([path(t.tree_, z, 2) for z in Zf])
    out.append(dict(rep=b, leaves=t.n_leaves(), ari=adjusted_rand_score(ref, lab), splits=json.dumps([(int(d), ZC[v], float(th)) for d, v, th in t.splits()])))
    pd.DataFrame(out).to_csv(F, index=False); print(b, t.n_leaves(), round(out[-1]['ari'], 3), [(d, ZC[v][2:], round(th, 3)) for d, v, th in t.splits()], round(time.time() - t0), flush=True)
