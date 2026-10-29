import numpy as np, pandas as pd, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sim7 import true_probs, regime, SHIFT
from sklearn.metrics import adjusted_rand_score
rows = []
for f in sorted(glob.glob(os.environ.get('SIM_OUT', '.') + '/data/*.npz')):
    pk = f.replace('.npz', '_py.pkl')
    if not os.path.exists(pk): continue
    b = os.path.basename(f)[:-4].split('_'); scen, m, design, rep = b[0], int(b[1]), b[2], int(b[3])
    a = np.load(f); Dn = pd.DataFrame(a['Dn'], columns=['unit', 's_prev', 'x1', 'x2', 'y']); u = Dn.unit.values.astype(int); rn = a['rn']; Zn = a['Zn']
    sh = SHIFT.get(scen, 0.0); T0 = true_probs(rn[u], np.zeros(len(u), int), Dn.x1.values, Dn.x2.values, sh); T1 = true_probs(rn[u], np.ones(len(u), int), Dn.x1.values, Dn.x2.values, sh)
    res = pd.read_pickle(pk); mc = f.replace('.npz', '_mob.csv')
    if os.path.exists(mc):
        M = pd.read_csv(mc); L = pd.read_csv(f.replace('.npz', '_mobleaf.csv')).set_index('unit').leaf
        nv = M.nvars.iloc[0]; vs = set() if (isinstance(nv, float) or nv == '') else {int(v[1:]) - 1 for v in str(nv).split(';')}
        res['MOB'] = dict(P0=M[['a0', 'a1', 'a2']].values, P1=M[['b0', 'b1', 'b2']].values, leaf=L.reindex(np.arange(len(Zn))).values, leaves=int(M.leaves.iloc[0]), vars=vs)
    s = Dn.s_prev.values.astype(int); y = Dn.y.values.astype(int)
    for nm, r in res.items():
        P = np.where(s[:, None] == 0, r['P0'], r['P1']); ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean()
        rmse = lambda A, B: float(np.sqrt(np.mean((A - B) ** 2)))
        rows.append(dict(scen=scen, m=m, design=design, rep=rep, method=nm, rmse_entry=rmse(r['P0'][:, 1], T0[:, 1]), rmse_exit0=rmse(r['P0'][:, 2], T0[:, 2]),
                         rmse_recovery=rmse(r['P1'][:, 0], T1[:, 0]), rmse_exit1=rmse(r['P1'][:, 2], T1[:, 2]), logloss=ll,
                         ari=adjusted_rand_score(rn, r['leaf']) if scen != 'null' else np.nan, leaves=r['leaves'],
                         spurious=len(set(r['vars']) - ({0, 1} if scen != 'null' else set()))))
R = pd.DataFrame(rows); R.to_csv(os.environ.get('SIM_OUT', '.') + '/sim7_raw.csv', index=False)
print(R.groupby(['scen', 'design', 'm', 'method'])[['rmse_entry', 'rmse_recovery', 'rmse_exit1', 'ari', 'leaves', 'spurious']].mean().round(3).to_string())
print(R[R.scen == 'null'].assign(split=lambda d: d.leaves > 1).groupby(['design', 'm', 'method']).split.agg(['mean', 'size']).round(3).to_string())
