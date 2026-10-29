"""Regime summaries of the full-data three-state tree, with bootstrap intervals over firms within each regime."""
import numpy as np, pandas as pd, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sabi3_design import design
import os; OFF = os.environ.get('OFF') == '1'; SUF = '_off' if OFF else ''
t = pickle.load(open(os.environ.get('OUT_DIR', 'out') + f'/tree_full{SUF}.pkl', 'rb'))
D = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/trans3.pkl'); ZC = [c for c in D.columns if c.startswith('z_')]
X1, X2 = design(D); Z = D[ZC].values.astype(float); P = t.predict_proba(Z, X1, X2, offset=np.load(os.environ.get('OUT_DIR', 'out') + '/offset_full.npy') if OFF else None)
def rule(node, path, out):
    if 'var' not in node: out[id(node)] = ' & '.join(path); return
    v = ZC[node['var']]; th = node['thr']
    rule(node['left'], path + [f'{v} <= {th:.4g}'], out); rule(node['right'], path + [f'{v} > {th:.4g}'], out)
rules = {}; rule(t.tree_, [], rules)
D['leaf'] = [rules[id(t._leaf(z))] for z in Z]; D[['p0', 'p1', 'p2']] = P
def summ(d):
    h = d[d.s_prev == 0]; s = d[d.s_prev == 1]
    p01, p02 = h.p1.mean(), h.p2.mean(); p10, p12 = s.p0.mean(), s.p2.mean()
    Q = np.array([[1 - p01 - p02, p01], [p10, 1 - p10 - p12]]); M = np.zeros((3, 3)); M[:2, :2] = Q; M[0, 2] = p02; M[1, 2] = p12; M[2, 2] = 1
    N = np.linalg.inv(np.eye(2) - Q); M10 = np.linalg.matrix_power(M, 10)
    return dict(p01=p01, p02=p02, p10=p10, p12=p12, exit_share=p12 / (p10 + p12), stress_spell=1 / (p10 + p12),
                years_to_exit_h=N[0].sum(), exit10_h=M10[0, 2], exit10_s=M10[1, 2],
                obs_p01=(h.y == 1).mean(), obs_p02=(h.y == 2).mean(), obs_p10=(s.y == 0).mean(), obs_p12=(s.y == 2).mean())
rng = np.random.default_rng(7); rows = []
for leaf, d in D.groupby('leaf'):
    base = summ(d); fs = d.firm.unique(); gi = {f: ix for f, ix in d.groupby('firm').indices.items()}; B = []
    for b in range(200):
        pick = rng.choice(fs, len(fs)); ix = np.concatenate([gi[f] for f in pick]); B.append(summ(d.iloc[ix]))
    B = pd.DataFrame(B); row = dict(regime=leaf, firms=len(fs), transitions=len(d), **base)
    for c in ['p01', 'p02', 'p10', 'p12', 'exit_share', 'stress_spell', 'years_to_exit_h', 'exit10_h', 'exit10_s']:
        row[c + '_lo'], row[c + '_hi'] = B[c].quantile([.025, .975])
    rows.append(row)
R = pd.DataFrame(rows).sort_values('exit10_h'); R.to_csv(os.environ.get('OUT_DIR', 'out') + f'/regimes3{SUF}.csv', index=False)
pd.set_option('display.width', 250); print(R[['regime', 'firms', 'p01', 'p02', 'p10', 'p12', 'exit_share', 'stress_spell', 'exit10_h', 'exit10_s', 'obs_p02', 'obs_p12']].round(3).to_string())
# coefficients
pd.DataFrame({rules[id(n)]: n['beta'] for n in [t._leaf(z) for z in np.unique(Z, axis=0)[:0]]})
def leaves(n):
    return [n] if 'var' not in n else leaves(n['left']) + leaves(n['right'])
C = pd.DataFrame({rules[id(n)]: n['beta'] for n in leaves(t.tree_)}).T; C.to_csv(os.environ.get('OUT_DIR', 'out') + f'/coef3{SUF}.csv')
