"""Simulation for three-state transition-regime trees (0 healthy, 1 adverse, 2 absorbing exit).
Regimes as in the two-state study (z1, z2). From state k the next state is multinomial logistic with reference
outcome 0 and linear predictors eta1 (adverse) and eta2 (exit) in (1, x1, x2); a unit effect eta_u ~ N(0, 0.5^2) is
added to eta1 only (misspecification). Units are followed for T periods or until exit.
Usage: python sim7.py gen REPS_REG REPS_NULL ; python sim7.py fit ; python sim7.py eval"""
import numpy as np, pandas as pd, sys, os, zlib, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from trtrees.multistate import MultiStateTree
from sklearn.metrics import adjusted_rand_score
from scipy.stats import norm
from joblib import Parallel, delayed
DIR = os.environ.get('SIM_DIR', 'simdata'); Q = 10; TAU = 0.5; CB = norm.ppf(1 - 0.05 / Q)
# coefficients (intercept, x1, x2) for eta1 and eta2 from each origin state, by regime
C = {0: dict(a1=(-2.5, .5, .2), a2=(-4.5, .3, 0), b1=(.5, .5, 0), b2=(-2.5, .3, 0)),
     1: dict(a1=(-1.5, 1., .2), a2=(-4.0, .3, 0), b1=(1., .8, 0), b2=(-2.0, .5, 0)),
     2: dict(a1=(-1.0, .5, .6), a2=(-3.5, .5, 0), b1=(2., .3, .3), b2=(-0.5, .8, 0))}
GH = np.polynomial.hermite.hermgauss(15)
def regime(Z, scen): return np.full(len(Z), 1) if scen == 'null' else np.where(Z[:, 0] <= .5, 0, np.where(Z[:, 1] <= .5, 1, 2))
SHIFT = {'rare': -1.0}                    # exit intercepts lowered by one in the rare-exit scenario
def lin(r, s, x1, x2, shift=0.0):
    A = lambda key, i: np.array([C[k][key][i] for k in range(3)])[r]
    e1 = np.where(s == 0, A('a1', 0) + A('a1', 1) * x1 + A('a1', 2) * x2, A('b1', 0) + A('b1', 1) * x1 + A('b1', 2) * x2)
    e2 = np.where(s == 0, A('a2', 0) + A('a2', 1) * x1 + A('a2', 2) * x2, A('b2', 0) + A('b2', 1) * x1 + A('b2', 2) * x2)
    return e1, e2 + shift
def sm(e1, e2): m = np.maximum(0, np.maximum(e1, e2)); a = np.stack([np.exp(-m), np.exp(e1 - m), np.exp(e2 - m)], -1); return a / a.sum(-1, keepdims=True)
def true_probs(r, s, x1, x2, shift=0.0):
    e1, e2 = lin(r, s, x1, x2, shift); nodes, w = GH
    return (w[None, :, None] * sm(e1[:, None] + np.sqrt(2) * TAU * nodes[None], np.broadcast_to(e2[:, None], (len(e1), len(nodes))))).sum(1) / np.sqrt(np.pi)
def simulate(scen, m, design, rng, burn=5):
    Z = rng.uniform(size=(m, Q)); r = regime(Z, scen); T = np.full(m, 10) if design == 'balanced' else rng.integers(3, 15, m)
    mu = .5 * rng.normal(size=m); eta = TAU * rng.normal(size=m); rows = []
    for u in range(m):
        n = T[u] + burn + 1; x1 = np.empty(n); x1[0] = mu[u] + rng.normal()
        for t in range(1, n): x1[t] = mu[u] + .7 * (x1[t - 1] - mu[u]) + np.sqrt(.51) * rng.normal()
        x2 = rng.normal(size=n); s = int(rng.random() < .3); t = 1
        while t < n:
            e1, e2 = lin(np.array([r[u]]), np.array([s]), x1[t:t + 1], x2[t:t + 1], SHIFT.get(scen, 0.0)); p = sm(e1 + eta[u], e2)[0]
            if t <= burn: p = sm(e1 + eta[u], np.array([-50.]))[0]          # no exit during burn-in
            new = int(rng.choice(3, p=p))
            if t > burn: rows.append((u, s, x1[t], x2[t], new))
            if new == 2: break
            s = new; t += 1
    return Z, r, pd.DataFrame(rows, columns=['unit', 's_prev', 'x1', 'x2', 'y'])
def design_matrix(D):
    a = (D.s_prev.values == 0).astype(float); b = 1 - a; B = np.column_stack([np.ones(len(D)), D.x1, D.x2])
    X = np.column_stack([B * a[:, None], B * b[:, None]]); return X, X.copy()
def gen(scen, m, design, rep):
    key = f'{scen}_{m}_{design}_{rep}'; f = f'{DIR}/{key}.npz'
    if os.path.exists(f): return
    rng = np.random.default_rng(zlib.crc32(key.encode())); Z, r, D = simulate(scen, m, design, rng)
    Zn, rn, Dn = simulate(scen, 1000, design, rng)
    # evaluation transitions for new units: their observed path plus counterfactual origin states for the probabilities
    np.savez(f, Z=Z, r=r, D=D.values, Zn=Zn, rn=rn, Dn=Dn.values)
class Oracle:
    def __init__(s, scen): s.scen = scen
    def fit(s, Z, r, D):
        X1, X2 = design_matrix(D); s.m = {}
        for k in np.unique(r):
            i = r[D.unit.values] == k; s.m[k] = MultiStateTree(max_depth=0).fit(np.zeros((i.sum(), 1)), X1[i], X2[i], D.y.values[i], D.unit.values[i])
        return s
    def proba(s, Zr, X1, X2):
        rr = regime(Zr, s.scen); out = np.empty((len(X1), 3))
        for k, mk in s.m.items(): i = rr == k; out[i] = mk.predict_proba(np.zeros((i.sum(), 1)), X1[i], X2[i])
        return out
def fit_one(f):
    out = f.replace('.npz', '_py.pkl')
    if os.path.exists(out): return
    a = np.load(f); Z, r = a['Z'], a['r']; D = pd.DataFrame(a['D'], columns=['unit', 's_prev', 'x1', 'x2', 'y']); D[['unit', 's_prev', 'y']] = D[['unit', 's_prev', 'y']].astype(int)
    Zn, rn = a['Zn'], a['rn']; Dn = pd.DataFrame(a['Dn'], columns=['unit', 's_prev', 'x1', 'x2', 'y']); Dn[['unit', 's_prev', 'y']] = Dn[['unit', 's_prev', 'y']].astype(int)
    scen = os.path.basename(f).split('_')[0]; rep = int(os.path.basename(f).split('_')[-1][:-4])
    X1, X2 = design_matrix(D); g = D.unit.values; Zo = Z[g]; y = D.y.values
    fits = {'global': MultiStateTree(max_depth=0).fit(Zo, X1, X2, y, g),
            'honest, 1-SE': MultiStateTree(criterion='honest', max_depth=4, min_units_leaf=10, se_rule=1, n_folds=10, random_state=rep).fit(Zo, X1, X2, y, g),
            'honest, Bonferroni': MultiStateTree(criterion='honest', max_depth=4, min_units_leaf=10, se_rule=CB, n_folds=10, random_state=rep).fit(Zo, X1, X2, y, g),
            'BIC': MultiStateTree(criterion='bic', max_depth=4, min_units_leaf=10).fit(Zo, X1, X2, y, g),
            'AIC': MultiStateTree(criterion='aic', max_depth=4, min_units_leaf=10).fit(Zo, X1, X2, y, g)}
    res = {}
    D0 = Dn.assign(s_prev=0); D1 = Dn.assign(s_prev=1); Zr = Zn[Dn.unit.values]
    for nm, t in fits.items():
        P0 = t.predict_proba(Zr, *design_matrix(D0)); P1 = t.predict_proba(Zr, *design_matrix(D1)); leaf = np.array([id(t._leaf(z)) for z in Zn])
        res[nm] = dict(P0=P0, P1=P1, leaf=leaf, leaves=t.n_leaves(), vars={v for _, v, _ in t.splits()})
    o = Oracle(scen).fit(Z, r, D); res['oracle partition'] = dict(P0=o.proba(Zr, *design_matrix(D0)), P1=o.proba(Zr, *design_matrix(D1)), leaf=regime(Zn, scen), leaves=len(o.m), vars=set())
    pd.to_pickle(res, out)
if __name__ == '__main__':
    if sys.argv[1] == 'gen':
        rr, rn = int(sys.argv[2]), int(sys.argv[3])
        jobs = [(s, m, d, r) for s, R in [('regimes', rr), ('null', rn)] for m in [100, 250, 500] for d in ['balanced', 'unbalanced'] for r in range(R)]
        if len(sys.argv) > 5 and sys.argv[5] == 'rare': jobs = [('rare', m, d, r) for m in [250, 500, 1000] for d in ['balanced', 'unbalanced'] for r in range(rr)]
        Parallel(n_jobs=int(sys.argv[4]) if len(sys.argv) > 4 else 1)(delayed(gen)(*j) for j in jobs); print('generated', len(jobs))
    elif sys.argv[1] == 'fit':
        fs = sorted(glob.glob(f'{DIR}/*.npz')); t0 = time.time()
        for k in range(0, len(fs), 20):
            Parallel(n_jobs=int(sys.argv[2]) if len(sys.argv) > 2 else 1)(delayed(fit_one)(f) for f in fs[k:k + 20]); print(k, round(time.time() - t0), flush=True)
