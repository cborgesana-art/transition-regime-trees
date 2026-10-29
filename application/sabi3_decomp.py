"""Decomposition of the gap between the honest tree and gradient boosting: nonlinear covariate effects (cubic splines)
without groups, the tree partition with spline leaves, and boosting without the fixed attributes."""
import numpy as np, pandas as pd, sys, os, pickle, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sabi3_design import design, XC
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
warnings.filterwarnings('ignore')
D = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/trans3.pkl'); ZC = [c for c in D.columns if c.startswith('z_')]
y = D.y.values.astype(int); g = D.firm.values; Z = D[ZC].values.astype(float); s = D.s_prev.values.astype(int)
fo = D.firm.map(pd.read_csv(os.environ.get('OUT_DIR', 'out') + '/folds.csv').set_index('firm').fold).values
CONT = ['roa', 'finl', 'log_liq', 'ebm', 'log_ta', 'log_age', 'zm', 'log_gap']; DUM = ['p_2011_13', 'p_2020']
which = sys.argv[1].split(','); X1, X2 = design(D)
def spline_fit(rows_tr, rows_te, P):
    """multinomial logistic with cubic splines (5 knots) of the continuous covariates, separately for each origin state"""
    for o in (0, 1):
        tr = rows_tr[s[rows_tr] == o]; te = rows_te[s[rows_te] == o]
        if len(te) == 0: continue
        st = SplineTransformer(n_knots=5, degree=3, knots='quantile', extrapolation='constant').fit(D[CONT].values[tr])
        f = lambda r: np.column_stack([st.transform(D[CONT].values[r]), D[DUM].values[r]])
        sc = StandardScaler().fit(f(tr)); A = sc.transform(f(tr)); B = sc.transform(f(te))
        cls = np.unique(y[tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(A, y[tr]); pr = m.predict_proba(B)
        out = np.full((len(te), 3), 1e-9); out[:, cls] = pr; P[te] = out / out.sum(1, keepdims=True)
for k in range(1, 6):
    tr = np.where(fo != k)[0]; te = np.where(fo == k)[0]
    for nm in which:
        out = os.environ.get('OUT_DIR', 'out') + f'/cv_{nm}_{k}.csv'
        if os.path.exists(out): continue
        t0 = time.time(); P = np.zeros((len(D), 3))
        if nm == 'gspline':
            spline_fit(tr, te, P)
        elif nm == 'tspline':
            t = pickle.load(open(os.environ.get('OUT_DIR', 'out') + f'/tree_honest_{k}.pkl', 'rb'))
            leaf = np.array([id(t._leaf(z)) for z in Z]); 
            for l in np.unique(leaf[tr]): spline_fit(tr[leaf[tr] == l], te[leaf[te] == l], P)
        elif nm == 'lgb_noz':
            F = np.column_stack([D[XC].values, s])
            m = lgb.LGBMClassifier(objective='multiclass', n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1)
            m.fit(F[tr], y[tr]); P[te] = m.predict_proba(F[te])
        P = P[te]
        pd.DataFrame(dict(firm=g[te], year=D.year.values[te], s_prev=s[te], y=y[te], p0=P[:, 0], p1=P[:, 1], p2=P[:, 2], leaves=np.nan)).to_csv(out, index=False)
        print(k, nm, round(time.time() - t0), flush=True)
