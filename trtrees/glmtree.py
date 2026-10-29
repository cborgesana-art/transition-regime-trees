"""Unit-transfer trees with generalized linear leaves (Poisson log link, logistic, Gaussian identity).

Each unit u contributes its mean per-observation loss
    l_u(beta) = (1/n_u) * sum_t [ b(eta_ut) - y_ut * eta_ut ] + const,   eta_ut = x_ut' beta + offset_ut,
(half the deviance), so every unit has the same weight. Leaf coefficients minimise sum_u l_u(beta) + (ridge/2)||beta||^2.

Split search. At a node with fit beta_S, every unit is summarised by (l_u, g_u, H_u), its loss, gradient and
Hessian at beta_S. The second-order expansion of l_u around beta_S has exactly the form used by the linear
tree, so all thresholds of an attribute are scanned with cumulative sums (one Newton step from beta_S for
every candidate child). The search is only used to choose the threshold.

Honest criterion. For every fold h of units (K folds, default 10): the parent model and, for every attribute,
the threshold search and the two child models are fitted on the units outside h (the child models by full
IRLS, warm-started at the Newton step), and the units in h are scored with their exact loss. This is the
honest criterion of the linear tree applied to a procedure that contains an approximate search and exact
child fits, so Proposition 1 of the note applies without change. The split is kept if the mean paired
difference (split minus no split) is below -se_rule times its standard error.
"""
import numpy as np


class _Family:
    def __init__(self, name):
        assert name in ("poisson", "binomial", "gaussian"); self.name = name

    def mean(self, eta):
        if self.name == "poisson": return np.exp(np.clip(eta, -30, 30))
        if self.name == "binomial": return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        return eta

    def var(self, mu):
        if self.name == "poisson": return mu
        if self.name == "binomial": return mu * (1 - mu)
        return np.ones_like(mu)

    def loss(self, y, eta):
        """Half unit deviance per observation (non-negative, zero at a perfect fit)."""
        mu = self.mean(eta)
        if self.name == "poisson":
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(y > 0, y * np.log(y / mu), 0.0)
            return t - (y - mu)
        if self.name == "binomial":
            mu = np.clip(mu, 1e-12, 1 - 1e-12)
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = np.where(y > 0, y * np.log(y / mu), 0.0); t0 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
            return t1 + t0
        return 0.5 * (y - mu) ** 2


class UnitTransferGLMTree:
    def __init__(self, family="poisson", criterion="honest", max_depth=4, min_units_leaf=5, ridge=1e-3,
                 se_rule=1.0, n_folds=10, random_state=0, max_features=None, max_iter=25, tol=1e-8):
        assert criterion in ("honest", "insample", "aic", "bic")
        self.family = family; self.fam = _Family(family); self.criterion = criterion; self.max_depth = max_depth
        self.min_units_leaf = min_units_leaf; self.ridge = ridge; self.se_rule = se_rule; self.n_folds = n_folds
        self.random_state = random_state; self.max_features = max_features; self.max_iter = max_iter; self.tol = tol
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------ data helpers
    def _obs(self, units):
        return np.concatenate([self.obs_[u] for u in units]) if len(units) else np.array([], int)

    def _irls(self, units, beta0=None):
        """Penalised unit-weighted GLM fit on the observations of the given units (Newton / IRLS)."""
        o = self._obs(units); X = self.X_[o]; y = self.y_[o]; w = self.w_[o]; off = self.off_[o]
        p = X.shape[1]; beta = np.zeros(p) if beta0 is None else beta0.copy()
        if beta0 is None and self.family == "poisson":
            beta[0] = np.log(max(np.average(y, weights=w), 1e-8)) - np.average(off, weights=w)
        if self.family == "gaussian": iters = 1
        else: iters = self.max_iter
        for _ in range(iters):
            eta = X @ beta + off; mu = self.fam.mean(eta)
            g = X.T @ (w * (mu - y)) + self.ridge * beta
            H = (X * (w * self.fam.var(mu))[:, None]).T @ X + self.ridge * np.eye(p)
            step = np.linalg.solve(H, g); beta = beta - step
            if np.max(np.abs(step)) < self.tol: break
        return beta

    def _dev(self, us, b):
        """Observation-level deviance (twice the summed negative log-likelihood) of the units in us under b."""
        return 2 * float(np.dot(self._unit_loss(us, b), [len(self.obs_[u]) for u in us]))

    def _unit_loss(self, units, beta):
        """Exact loss l_u(beta) for each unit in units (beta: (p,) or (len(units), p))."""
        out = np.empty(len(units)); B = np.atleast_2d(beta)
        for k, u in enumerate(units):
            o = self.obs_[u]; b = B[0] if len(B) == 1 else B[k]
            out[k] = self.fam.loss(self.y_[o], self.X_[o] @ b + self.off_[o]).mean()
        return out

    def _stats(self, units, beta):
        """Per-unit loss, gradient and Hessian at beta (unit-weighted)."""
        p = self.X_.shape[1]; m = len(units)
        L = np.empty(m); G = np.empty((m, p)); Hs = np.empty((m, p, p))
        for k, u in enumerate(units):
            o = self.obs_[u]; X = self.X_[o]; y = self.y_[o]; n = len(o)
            eta = X @ beta + self.off_[o]; mu = self.fam.mean(eta)
            L[k] = self.fam.loss(y, eta).mean(); G[k] = X.T @ (mu - y) / n
            Hs[k] = (X * (self.fam.var(mu) / n)[:, None]).T @ X
        return L, G, Hs

    def _sweep(self, L, G, Hs, beta):
        """Second-order losses of all left/right splits of an ordered set of units; Newton-step child betas."""
        p = G.shape[1]; I = self.ridge * np.eye(p)
        cH = np.cumsum(Hs, 0); cG = np.cumsum(G, 0); cL = np.cumsum(L)
        HL, GL, LL = cH[:-1], cG[:-1], cL[:-1]
        HR, GR, LR = cH[-1][None] - HL, cG[-1][None] - GL, cL[-1] - LL
        gl = GL + self.ridge * beta; gr = GR + self.ridge * beta
        dL = -np.linalg.solve(HL + I, gl[..., None])[..., 0]; dR = -np.linalg.solve(HR + I, gr[..., None])[..., 0]
        qL = LL + 0.5 * np.einsum("kp,kp->k", dL, gl); qR = LR + 0.5 * np.einsum("kp,kp->k", dR, gr)
        return qL + qR, beta + dL, beta + dR

    def _best_threshold(self, units, z, beta, L, G, Hs):
        order = np.argsort(z, kind="mergesort"); zs = z[order]; mk = len(order); mu = self.min_units_leaf
        q, bL, bR = self._sweep(L[order], G[order], Hs[order], beta)
        pos = np.arange(1, mk); valid = (pos >= mu) & (mk - pos >= mu) & (zs[pos - 1] < zs[pos])
        if not valid.any(): return None
        k = int(np.argmin(np.where(valid, q, np.inf)))
        return (zs[k] + zs[k + 1]) / 2, bL[k], bR[k], q[k]

    def _holdouts(self, m, depth, idx):
        if not self.n_folds or self.n_folds >= m:
            return [np.array([i]) for i in range(m)]
        rng = np.random.default_rng(self.random_state + 7919 * depth + int(idx[0]))
        perm = rng.permutation(m); return [perm[f::self.n_folds] for f in range(self.n_folds)]

    def _candidates(self):
        q = self.Z_.shape[1]
        if not self.max_features or self.max_features >= q: return range(q)
        return np.sort(self._rng.choice(q, self.max_features, replace=False))

    # ------------------------------------------------------------------ growing
    def _grow(self, idx, depth, beta0):
        beta = self._irls(idx, beta0); m = len(idx)
        node = {"units": idx, "beta": beta, "n_units": m}
        if depth >= self.max_depth or m < 2 * self.min_units_leaf:
            return node
        L, G, Hs = self._stats(idx, beta)
        if self.criterion in ("insample", "aic", "bic"):
            best = None
            for j in self._candidates():
                z = self.Z_[idx, j]
                if z.min() == z.max(): continue
                r = self._best_threshold(idx, z, beta, L, G, Hs)
                if r is not None and (best is None or r[3] < best[0]): best = (r[3], j, r[0])
            if best is None or best[0] >= L.sum(): return node
            j, thr = best[1], best[2]
            if self.criterion in ("aic", "bic"):
                # information-criterion stopping as in phase-type survival trees: exact child refits, observation-level
                # deviance, split kept if the deviance drop exceeds k * p (k = 2 for AIC, log(number of observations) for BIC)
                z = self.Z_[idx, j]; le = idx[z <= thr]; ri = idx[z > thr]
                dev = self._dev
                bl = self._irls(le, beta); br = self._irls(ri, beta)
                drop = dev(idx, beta) - dev(le, bl) - dev(ri, br)
                n_obs = sum(len(self.obs_[u]) for u in idx); p = len(beta)
                k = 2.0 if self.criterion == "aic" else np.log(n_obs)
                node["ic_drop"] = drop
                if drop <= k * p: return node
        else:
            H = self._holdouts(m, depth, idx)
            if m < 2 * self.min_units_leaf + max(len(h) for h in H): return node
            parent_u = np.empty(m); trains = []
            for h in H:
                keep = np.setdiff1d(np.arange(m), h); bp = self._irls(idx[keep], beta)
                parent_u[h] = self._unit_loss(idx[h], bp); trains.append((keep, bp))
            node["loss"] = float(parent_u.sum()); best = None
            for j in self._candidates():
                z = self.Z_[idx, j]
                if z.min() == z.max(): continue
                per = np.empty(m); ok = True
                for h, (keep, bp) in zip(H, trains):
                    r = self._best_threshold(idx[keep], z[keep], beta, L[keep], G[keep], Hs[keep])
                    if r is None: ok = False; break
                    thr, bl, br, _ = r; lk = keep[z[keep] <= thr]; rk = keep[z[keep] > thr]
                    bl = self._irls(idx[lk], bl); br = self._irls(idx[rk], br)
                    left = z[h] <= thr
                    per[h] = np.where(left, self._unit_loss(idx[h], bl), self._unit_loss(idx[h], br))
                if ok and (best is None or per.sum() < best[0]): best = (per.sum(), j, per.copy())
            if best is None or best[0] >= node["loss"]: return node
            d = best[2] - parent_u
            node.update(cand_mean=float(d.mean()), cand_se=float(d.std(ddof=1) / np.sqrt(m)), cand_var=int(best[1]))
            if d.mean() > -self.se_rule * d.std(ddof=1) / np.sqrt(m): return node
            j = best[1]; thr = self._best_threshold(idx, self.Z_[idx, j], beta, L, G, Hs)[0]
            node.update(honest_loss=best[0], mean_diff=d.mean(), se_diff=d.std(ddof=1) / np.sqrt(m))
        z = self.Z_[idx, j]; le = idx[z <= thr]; ri = idx[z > thr]
        node.update(var=j, thr=thr, left=self._grow(le, depth + 1, beta), right=self._grow(ri, depth + 1, beta))
        return node

    # ------------------------------------------------------------------ public API
    def fit(self, Z_unit, X, y, groups, offset=None):
        """Z_unit: (n_obs, q) unit attributes, constant within unit; X: (n_obs, p) with a column of ones;
        y: counts (poisson), 0/1 (binomial) or real (gaussian); offset: e.g. log exposure."""
        X = np.asarray(X, float); y = np.asarray(y, float); groups = np.asarray(groups); Z_unit = np.asarray(Z_unit, float)
        units, inv = np.unique(groups, return_inverse=True); m = len(units)
        order = np.argsort(inv, kind="mergesort"); starts = np.searchsorted(inv[order], np.arange(m + 1))
        self.obs_ = [order[starts[k]:starts[k + 1]] for k in range(m)]
        n_u = np.diff(starts); self.w_ = 1.0 / n_u[inv]
        self.X_, self.y_ = X, y; self.off_ = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
        Zu = np.stack([Z_unit[o[0]] for o in self.obs_])
        for k, o in enumerate(self.obs_):
            if not np.allclose(Z_unit[o], Zu[k]): raise ValueError("Z must be constant within each unit")
        self.units_, self.Z_ = units, Zu
        self.tree_ = self._grow(np.arange(m), 0, None)
        del self.X_, self.y_, self.off_, self.w_, self.obs_
        return self

    def _leaf(self, z):
        node = self.tree_
        while "var" in node:
            node = node["left"] if z[node["var"]] <= node["thr"] else node["right"]
        return node

    def coef(self, Z_new):
        return np.stack([self._leaf(z)["beta"] for z in np.asarray(Z_new, float)])

    def predict(self, Z_unit, X, offset=None):
        """Predicted mean (counts per unit of exposure times exp(offset), or probabilities)."""
        X = np.asarray(X, float); B = self.coef(Z_unit)
        eta = np.einsum("ip,ip->i", X, B) + (0 if offset is None else np.asarray(offset, float))
        return self.fam.mean(eta)

    def n_leaves(self, node=None):
        node = self.tree_ if node is None else node
        return 1 if "var" not in node else self.n_leaves(node["left"]) + self.n_leaves(node["right"])

    def splits(self, node=None, depth=0):
        node = self.tree_ if node is None else node
        if "var" not in node: return []
        return [(depth, node["var"], node["thr"])] + self.splits(node["left"], depth + 1) + self.splits(node["right"], depth + 1)


def glm_select_depth_grouped_cv(Z_unit, X, y, groups, family="poisson", depths=range(0, 5), n_folds=5, seed=0, offset=None, **kw):
    """Baseline: in-sample GLM trees with depth chosen by grouped K-fold CV (unit-weighted deviance)."""
    groups = np.asarray(groups); units = np.unique(groups); rng = np.random.default_rng(seed)
    fold = dict(zip(units, rng.permutation(len(units)) % n_folds)); f = np.array([fold[g] for g in groups])
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float); fam = _Family(family); err = []
    for d in depths:
        e = 0.0
        for k in range(n_folds):
            tr = f != k; te = ~tr
            t = UnitTransferGLMTree(family=family, criterion="insample", max_depth=d, **kw).fit(Z_unit[tr], X[tr], y[tr], groups[tr], off[tr])
            B = t.coef(Z_unit[te]); eta = np.einsum("ip,ip->i", X[te], B) + off[te]; l = fam.loss(y[te], eta)
            gte = groups[te]; _, ii = np.unique(gte, return_inverse=True)
            e += np.sum(np.bincount(ii, l) / np.bincount(ii))
        err.append(e)
    d_best = list(depths)[int(np.argmin(err))]
    return UnitTransferGLMTree(family=family, criterion="insample", max_depth=d_best, **kw).fit(Z_unit, X, y, groups, offset), d_best
