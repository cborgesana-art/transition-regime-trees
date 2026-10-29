"""Sensitivity of the regime summaries to the definition of exit, with the partition of the full-data tree fixed:
base; 'inactive' (inactive firms also exit); 'stopreport' (any firm, except merged ones, whose last known state is in
2022 or earlier exits in the following year); 'no2024' (transitions into 2024 removed). The spline offset and the
leaf models are refitted for each definition."""
import os
import numpy as np, pandas as pd, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sabi3_design as SD
from trtrees.multistate import MultiStateTree
from sabi3_offset import SplineOffset
EXIT0 = set(SD.EXIT)
def build(variant):
    d = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/wide.pkl'); L = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/long.pkl')
    st = d['Situação_Actual']
    last = L[L.state.notna()].groupby('firm').year.max()
    if variant == 'inactive': ex = st.isin(EXIT0 | {'Inactiva'})
    elif variant == 'stopreport': ex = st.isin(EXIT0) | ((last.reindex(d.index) <= 2022) & (st != 'Fusão'))
    else: ex = st.isin(EXIT0)
    SD.EXIT = set(); D = None
    # reuse the base builder with a firm-level exit indicator
    orig = d['Situação_Actual'].copy(); d['Situação_Actual'] = np.where(ex, 'EXIT', 'CENS')
    pd.to_pickle(d, 'wide_variant.pkl')
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sabi3_design.py')).read().replace("os.environ.get('SABI_DIR', 'data') + '/wide.pkl'", "'wide_variant.pkl'").replace("EXIT = {'Dissolução', 'Liquidação', 'Insolvência', 'Encerrada', 'Extinção'}", "EXIT = {'EXIT'}")
    ns = {'__name__': 'variant'}; exec(compile(src, 'variant', 'exec'), ns); D = ns['build']()
    if variant == 'no2024': D = D[D.year <= 2023].reset_index(drop=True)
    return D, ns['design']
t = pickle.load(open(os.environ.get('OUT_DIR', 'out') + '/tree_full_off.pkl', 'rb'))
lab = pd.read_csv(os.environ.get('OUT_DIR', 'out') + '/regimes3_off_labelled.csv').set_index('regime').label
def rules(node, path, out):
    if 'var' not in node: out[id(node)] = ' & '.join(path); return
    v = ZC[node['var']]; th = node['thr']; rules(node['left'], path + [f'{v} <= {th:.4g}'], out); rules(node['right'], path + [f'{v} > {th:.4g}'], out)
rows = []
for variant in ['base', 'inactive', 'stopreport', 'no2024']:
    D, design = build(variant if variant != 'no2024' else 'base') if variant != 'no2024' else build('no2024')
    ZC = [c for c in D.columns if c.startswith('z_')]; X1, X2 = design(D); Z = D[ZC].values.astype(float); y = D.y.values; g = D.firm.values
    R = {}; rules(t.tree_, [], R); leaf = np.array([R[id(t._leaf(z))] for z in Z])
    rr = np.arange(len(D)); O = SplineOffset().fit(D, rr).offset(D, rr)
    tot = dict(variant=variant, regime='all', exits=int((y == 2).sum()), p02=(y[D.s_prev.values == 0] == 2).mean(), p12=(y[D.s_prev.values == 1] == 2).mean()); rows.append(tot)
    for l in np.unique(leaf):
        i = leaf == l; m = MultiStateTree(max_depth=0, max_iter=25).fit(Z[i], X1[i], X2[i], y[i], g[i], offset=O[i]); P = m.predict_proba(Z[i], X1[i], X2[i], offset=O[i])
        s = D.s_prev.values[i]; p01, p02 = P[s == 0, 1].mean(), P[s == 0, 2].mean(); p10, p12 = P[s == 1, 0].mean(), P[s == 1, 2].mean()
        Q = np.array([[1 - p01 - p02, p01, p02], [p10, 1 - p10 - p12, p12], [0, 0, 1]]); M10 = np.linalg.matrix_power(Q, 10)
        rows.append(dict(variant=variant, regime=lab.get(l, l), firms=len(np.unique(g[i])), p02=p02, p12=p12, exit_share=p12 / (p10 + p12), exit10_h=M10[0, 2]))
    print(variant, tot, flush=True)
O = pd.DataFrame(rows); O.to_csv(os.environ.get('OUT_DIR', 'out') + '/censor3.csv', index=False)
W = O[O.regime != 'all'].pivot(index='regime', columns='variant', values=['exit_share', 'exit10_h'])
print(W.round(3).to_string())
B = O[(O.regime != 'all') & (O.variant == 'base')].set_index('regime')
for v in ['inactive', 'stopreport', 'no2024']:
    V = O[(O.regime != 'all') & (O.variant == v)].set_index('regime').reindex(B.index)
    print(v, 'spearman exit_share', round(B.exit_share.corr(V.exit_share, method='spearman'), 3), 'exit10_h', round(B.exit10_h.corr(V.exit10_h, method='spearman'), 3),
          'max abs diff exit10', round((B.exit10_h - V.exit10_h).abs().max(), 3), 'ratio', round((V.exit10_h / B.exit10_h).min(), 2), round((V.exit10_h / B.exit10_h).max(), 2))
