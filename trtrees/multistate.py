"""Transition-regime trees for a three-state process: 0 healthy, 1 adverse, 2 absorbing exit.

Within a leaf, a transition from state k in {0, 1} has a multinomial logistic distribution over {0, 1, 2}, with
outcome 0 as reference, linear predictor eta_1 = x1' b1 for state 1 and eta_2 = x2' b2 for exit. The designs x1 and
x2 are interacted with the previous state (see real3/sabi3_design.py), so each origin state has its own coefficients.
The unit loss is the mean negative log-likelihood per transition, as in the two-state case, and the tree machinery
(second-order threshold search, honest unit-level criterion, AIC/BIC stopping) is inherited unchanged."""
import numpy as np
from .glmtree import UnitTransferGLMTree


def _probs(E1, E2):
    m = np.maximum(0, np.maximum(E1, E2)); a0 = np.exp(-m); a1 = np.exp(E1 - m); a2 = np.exp(E2 - m); s = a0 + a1 + a2
    return a0 / s, a1 / s, a2 / s


class MultiStateTree(UnitTransferGLMTree):
    def __init__(self, criterion="honest", chunk=4000, unit_loss="sum", **kw):
        """unit_loss: "sum" (negative log-likelihood of the unit's trajectory, so the leaf fit is the ordinary maximum
        likelihood estimate) or "mean" (mean per transition, every unit with equal weight)."""
        super().__init__(family="binomial", criterion=criterion, **kw); self.chunk = chunk; self.unit_loss = unit_loss

    # observations: X_ = [X1 | X2], y_ in {0,1,2}
    def _parts(self, o, beta):
        X = self.X_[o]; X1, X2 = X[:, :self.P1], X[:, self.P1:]
        E1 = X1 @ beta[:self.P1] + self.O_[o, 0]; E2 = X2 @ beta[self.P1:] + self.O_[o, 1]
        return X1, X2, _probs(E1, E2)

    def _nll(self, y, p0, p1, p2):
        p = np.where(y == 0, p0, np.where(y == 1, p1, p2)); return -np.log(np.clip(p, 1e-300, None))

    def _irls(self, units, beta0=None):
        o = self._obs(units); y = self.y_[o]; w = self.w_[o] if self.unit_loss == 'mean' else np.ones(len(o)); P = self.X_.shape[1]
        beta = np.zeros(P) if beta0 is None else beta0.copy()
        for _ in range(self.max_iter):
            X1, X2, (p0, p1, p2) = self._parts(o, beta)
            r1 = p1 - (y == 1); r2 = p2 - (y == 2)
            g = np.concatenate([X1.T @ (w * r1), X2.T @ (w * r2)]) + self.ridge * beta
            H11 = (X1 * (w * p1 * (1 - p1))[:, None]).T @ X1; H22 = (X2 * (w * p2 * (1 - p2))[:, None]).T @ X2
            H12 = -(X1 * (w * p1 * p2)[:, None]).T @ X2
            H = np.block([[H11, H12], [H12.T, H22]]) + self.ridge * np.eye(P)
            step = np.linalg.solve(H, g)
            f0 = self._obj(o, y, w, beta); lam = 1.0
            while lam > 1e-4 and self._obj(o, y, w, beta - lam * step) > f0 + 1e-12: lam /= 2      # step halving
            beta = beta - lam * step
            if np.max(np.abs(lam * step)) < self.tol: break
        return beta

    def _obj(self, o, y, w, beta):
        X1, X2, pr = self._parts(o, beta); return float(w @ self._nll(y, *pr)) + 0.5 * self.ridge * beta @ beta

    def _unit_loss(self, units, beta):
        out = np.empty(len(units)); B = np.atleast_2d(beta)
        if len(B) == 1:
            o = self._obs(units); X1, X2, pr = self._parts(o, B[0]); l = self._nll(self.y_[o], *pr)
            n = np.array([len(self.obs_[u]) for u in units]); starts = np.concatenate([[0], np.cumsum(n)[:-1]])
            return np.add.reduceat(l, starts) / (n if self.unit_loss == 'mean' else 1)
        for k, u in enumerate(units):
            o = self.obs_[u]; X1, X2, pr = self._parts(o, B[k]); l = self._nll(self.y_[o], *pr); out[k] = l.mean() if self.unit_loss == 'mean' else l.sum()
        return out

    def _stats(self, units, beta):
        P = self.X_.shape[1]; m = len(units); L = np.empty(m); G = np.empty((m, P)); Hs = np.empty((m, P, P))
        for c in range(0, m, self.chunk):
            us = units[c:c + self.chunk]; o = self._obs(us); y = self.y_[o]
            n = np.array([len(self.obs_[u]) for u in us]); starts = np.concatenate([[0], np.cumsum(n)[:-1]]); wn = np.repeat(1.0 / n, n) if self.unit_loss == 'mean' else np.ones(len(o))
            X1, X2, (p0, p1, p2) = self._parts(o, beta)
            L[c:c + len(us)] = np.add.reduceat(self._nll(y, p0, p1, p2) * wn, starts)
            g = np.column_stack([X1 * ((p1 - (y == 1)) * wn)[:, None], X2 * ((p2 - (y == 2)) * wn)[:, None]])
            G[c:c + len(us)] = np.add.reduceat(g, starts, axis=0)
            # Hessian of -log p: blocks X1' diag(p1(1-p1)) X1, -X1' diag(p1 p2) X2, X2' diag(p2(1-p2)) X2
            A = np.column_stack([X1 * np.sqrt(p1 * (1 - p1) * wn)[:, None], np.zeros_like(X2)])
            Hc = np.empty((len(o), P, P))
            Hc[:, :self.P1, :self.P1] = np.einsum('ni,nj->nij', X1, X1 * (p1 * (1 - p1) * wn)[:, None])
            Hc[:, self.P1:, self.P1:] = np.einsum('ni,nj->nij', X2, X2 * (p2 * (1 - p2) * wn)[:, None])
            C = -np.einsum('ni,nj->nij', X1, X2 * (p1 * p2 * wn)[:, None]); Hc[:, :self.P1, self.P1:] = C; Hc[:, self.P1:, :self.P1] = C.transpose(0, 2, 1)
            Hs[c:c + len(us)] = np.add.reduceat(Hc, starts, axis=0); del Hc, C, A
        return L, G, Hs

    def _best_threshold(self, units, z, beta, L, G, Hs, kappas=(0.0, 0.1, 1.0)):
        """Shortlist thresholds with the second-order expansion, undamped and with Levenberg-Marquardt damping
        kappa * diag(H) of each child's Hessian, then refit the children of every shortlisted threshold exactly and
        return the one with the smallest exact training loss. The undamped expansion alone can favour very unequal
        splits whose gain it overstates, because a rare outcome makes the Hessian of a small child nearly singular."""
        order = np.argsort(z, kind="mergesort"); zs = z[order]; mk = len(order); mu = self.min_units_leaf
        pos = np.arange(1, mk); valid = (pos >= mu) & (mk - pos >= mu) & (zs[pos - 1] < zs[pos])
        if not valid.any(): return None
        P = G.shape[1]; I = self.ridge * np.eye(P)
        cH = Hs[order]; np.cumsum(cH, axis=0, out=cH)
        cG = np.cumsum(G[order], 0); cL = np.cumsum(L[order]); HT, GT, LT = cH[-1].copy(), cG[-1].copy(), cL[-1]
        cand = np.where(valid)[0]; best = {kp: (np.inf, None) for kp in kappas}
        for c in range(0, len(cand), self.chunk):
            k = cand[c:c + self.chunk]
            HL = cH[k]; HR = HT[None] - HL; gl = cG[k] + self.ridge * beta; gr = (GT - cG[k]) + self.ridge * beta
            dgL = np.einsum('kii->ki', HL); dgR = np.einsum('kii->ki', HR)
            for kp in kappas:
                AL = HL + I; AR = HR + I
                if kp:
                    AL = AL + kp * dgL[:, :, None] * np.eye(P); AR = AR + kp * dgR[:, :, None] * np.eye(P)
                dL = -np.linalg.solve(AL, gl[..., None])[..., 0]; dR = -np.linalg.solve(AR, gr[..., None])[..., 0]
                q = cL[k] + 0.5 * np.einsum('kp,kp->k', dL, gl) + (LT - cL[k]) + 0.5 * np.einsum('kp,kp->k', dR, gr)
                i = int(np.argmin(q))
                if q[i] < best[kp][0]: best[kp] = (q[i], (k[i], beta + dL[i], beta + dR[i]))
        del cH
        out = None
        for kp, (q, info) in best.items():
            if info is None: continue
            k, bl0, br0 = info; thr = (zs[k] + zs[k + 1]) / 2
            if out is not None and thr == out[0]: continue
            le = units[z <= thr]; ri = units[z > thr]; bl = self._irls(le, bl0); br = self._irls(ri, br0)
            loss = float(self._unit_loss(le, bl).sum() + self._unit_loss(ri, br).sum())
            if out is None or loss < out[3]: out = (thr, bl, br, loss)
        return out

    def _dev(self, us, b):
        L = self._unit_loss(us, b)
        return 2 * float(L.sum() if self.unit_loss == 'sum' else np.dot(L, [len(self.obs_[u]) for u in us]))

    def fit(self, Z_unit, X1, X2, y, groups, offset=None):
        """offset: optional (n, 2) array added to the linear predictors of the adverse and exit outcomes, for
        instance those of a global model with nonlinear covariate effects; the leaves then model deviations from it."""
        self.P1 = X1.shape[1]; self.O_ = np.zeros((len(y), 2)) if offset is None else np.asarray(offset, float)
        return super().fit(Z_unit, np.column_stack([X1, X2]), y, groups)

    def predict_proba(self, Z_unit, X1, X2, offset=None):
        O = np.zeros((len(X1), 2)) if offset is None else np.asarray(offset, float)
        B = self.coef(Z_unit); E1 = np.einsum('ip,ip->i', X1, B[:, :self.P1]) + O[:, 0]; E2 = np.einsum('ip,ip->i', X2, B[:, self.P1:]) + O[:, 1]
        return np.column_stack(_probs(E1, E2))
