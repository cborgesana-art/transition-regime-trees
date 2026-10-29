"""Global three-state model with cubic-spline covariate effects, used as an offset for the leaves of the tree."""
import numpy as np
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
CONT = ['roa', 'finl', 'log_liq', 'ebm', 'log_ta', 'log_age', 'zm', 'log_gap']; DUM = ['p_2011_13', 'p_2020']
class SplineOffset:
    def fit(self, D, rows):
        self.m = {}
        for o in (0, 1):
            r = rows[D.s_prev.values[rows] == o]
            st = SplineTransformer(n_knots=5, degree=3, knots='quantile', extrapolation='constant').fit(D[CONT].values[r])
            A = np.column_stack([st.transform(D[CONT].values[r]), D[DUM].values[r]]); sc = StandardScaler().fit(A)
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(A), D.y.values[r].astype(int)); self.m[o] = (st, sc, lr)
        return self
    def offset(self, D, rows):
        """(len(rows), 2) linear predictors of the adverse and exit outcomes relative to the healthy state"""
        O = np.zeros((len(rows), 2)); s = D.s_prev.values[rows]
        for o in (0, 1):
            i = np.where(s == o)[0]; r = rows[i]
            if not len(i): continue
            st, sc, lr = self.m[o]; A = sc.transform(np.column_stack([st.transform(D[CONT].values[r]), D[DUM].values[r]]))
            L = A @ lr.coef_.T + lr.intercept_          # class scores for classes lr.classes_ (0, 1, 2)
            O[i, 0] = L[:, 1] - L[:, 0]; O[i, 1] = L[:, 2] - L[:, 0]
        return O
