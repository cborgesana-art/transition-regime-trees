## MOB (partykit) with a multinomial leaf model and firm-clustered score tests, and multistate Cox models
## (cause-specific Cox per transition, gap time) on the same five folds. Args: method (mob|cox)
suppressMessages({library(partykit); library(survival); library(data.table)})
args <- commandArgs(TRUE); meth <- args[1]; test <- length(args) > 1
D <- fread(file.path(Sys.getenv('SABI_DIR', 'data'), 'trans3.csv')); fo <- fread(file.path(Sys.getenv('OUT_DIR', 'out'), 'folds.csv')); D <- merge(D, fo, by = 'firm'); setorder(D, firm, year)
XC <- c('roa','finl','log_liq','ebm','log_ta','log_age','zm','zm2','p_2011_13','p_2020','log_gap'); XE <- c('roa','finl','log_liq','log_ta','log_age','zm','log_gap')
ZC <- grep('^z_', names(D), value = TRUE)
if (test) { set.seed(1); D <- D[firm %in% sample(unique(firm), 4000)] }
a <- as.numeric(D$s_prev == 0); b <- 1 - a
B1 <- cbind(1, as.matrix(D[, ..XC])); B2 <- cbind(1, as.matrix(D[, ..XE]))
X <- cbind(B1 * a, B1 * b, B2 * a, B2 * b); P1 <- 2 * ncol(B1); colnames(X) <- paste0('w', seq_len(ncol(X)))
softmax <- function(E1, E2) { m <- pmax(0, E1, E2); A <- cbind(exp(-m), exp(E1 - m), exp(E2 - m)); A / rowSums(A) }
mnfit <- function(y, x, start = NULL, weights = NULL, offset = NULL, ..., estfun = FALSE, object = FALSE) {
  P <- ncol(x); beta <- if (is.null(start)) rep(0, P) else start; x1 <- x[, 1:P1, drop = FALSE]; x2 <- x[, (P1 + 1):P, drop = FALSE]
  for (it in 1:25) {
    pr <- softmax(drop(x1 %*% beta[1:P1]), drop(x2 %*% beta[(P1 + 1):P]))
    r1 <- pr[, 2] - (y == 1); r2 <- pr[, 3] - (y == 2)
    g <- c(crossprod(x1, r1), crossprod(x2, r2)) + 1e-3 * beta
    H11 <- crossprod(x1 * (pr[, 2] * (1 - pr[, 2])), x1); H22 <- crossprod(x2 * (pr[, 3] * (1 - pr[, 3])), x2); H12 <- -crossprod(x1 * (pr[, 2] * pr[, 3]), x2)
    H <- rbind(cbind(H11, H12), cbind(t(H12), H22)) + diag(1e-3, P)
    st <- solve(H, g); beta <- beta - st; if (max(abs(st)) < 1e-8) break
  }
  pr <- softmax(drop(x1 %*% beta[1:P1]), drop(x2 %*% beta[(P1 + 1):P])); names(beta) <- colnames(x)
  ll <- sum(log(pmax(pr[cbind(seq_along(y), y + 1)], 1e-300)))
  ef <- if (estfun) cbind(x1 * ((y == 1) - pr[, 2]), x2 * ((y == 2) - pr[, 3])) else NULL
  ## columns that are identically zero in a node (a period indicator or covariate block absent there) make the
  ## clustered score covariance singular; a negligible perturbation keeps the Cholesky factorisation defined
  if (estfun) ef <- ef + matrix(rnorm(length(ef), sd = 1e-9), nrow(ef))
  list(coefficients = beta, objfun = -ll, estfun = ef, object = if (object) list(beta = beta) else NULL)
}
for (k in 1:5) {
  out <- sprintf(if (test) 't_%s_%d.csv' else file.path(Sys.getenv('OUT_DIR', 'out'), 'cv_%s_%d.csv'), meth, k); if (file.exists(out)) next
  tr <- D$fold != k; te <- D$fold == k; t0 <- Sys.time()
  if (meth == 'mob') {
    ## partitioning variables discretised to at most 20 quantile values, so that the exhaustive split-point search of mob
    ## refits the leaf model at most 19 times per node
    for (zc in ZC) { v <- D[[zc]]; if (length(unique(v)) > 20) { br <- unique(quantile(v, seq(0.05, 1, 0.05), na.rm = TRUE)); D[[paste0(zc, '_b')]] <- br[findInterval(v, br, left.open = TRUE) + 1] } else D[[paste0(zc, '_b')]] <- v }
    ZB <- paste0(ZC, '_b')
    dd <- data.frame(y = D$y[tr], X[tr, ], setNames(D[tr, ..ZB], ZC)); dd$firm <- D$firm[tr]
    fml <- as.formula(paste('y ~', paste(colnames(X), collapse = '+'), '- 1 |', paste(ZC, collapse = '+')))
    m <- mob(fml, data = dd, fit = mnfit, cluster = firm, control = mob_control(maxdepth = 5, minsize = if (test) 300 else 2400, alpha = 0.05, bonferroni = TRUE, ytype = 'vector', xtype = 'matrix'))
    nd <- data.frame(y = D$y[te], X[te, ], setNames(D[te, ..ZB], ZC)); nid <- predict(m, newdata = nd, type = 'node')
    cf <- coef(m); if (is.null(dim(cf))) cf <- matrix(cf, nrow = 1, dimnames = list(names(nodeids(m, terminal = TRUE)), names(cf)))
    Bn <- cf[as.character(nid), , drop = FALSE]; Xt <- X[te, ]
    P <- softmax(rowSums(Xt[, 1:P1] * Bn[, 1:P1]), rowSums(Xt[, (P1 + 1):ncol(X)] * Bn[, (P1 + 1):ncol(X)])); leaves <- length(nodeids(m, terminal = TRUE))
  } else {
    ## cause-specific Cox models for the four transitions out of states 0 and 1, gap time in the current state,
    ## counting-process form, stratified by episode number for transitions out of stress (capped at 3)
    D[, ep3 := pmin(episode, 3)]; feats <- c(setdiff(XC, 'log_gap'), ZC); H <- matrix(0, sum(te), 3); haz <- list()
    for (o in 0:1) for (j in setdiff(0:2, o)) {
      dtr <- D[tr & D$s_prev == o]; dtr[, ev := as.integer(y == j)]
      f <- as.formula(paste('Surv(gap - 1, gap, ev) ~', paste(feats, collapse = '+'), if (o == 1) '+ strata(ep3)' else ''))
      cm <- coxph(f, data = dtr, ties = 'breslow', control = coxph.control(iter.max = 10))
      dte <- D[te]; ix <- which(dte$s_prev == o)
      bh <- basehaz(cm, centered = FALSE); setDT(bh)
      if (o == 1) { bh[, d := c(hazard[1], diff(hazard)), by = strata]; bh[, ep3 := as.integer(sub('ep3=', '', strata))] } else { bh[, d := c(hazard[1], diff(hazard))]; bh[, ep3 := 0L] }
      bh <- bh[, .(d = sum(d)), by = .(ep3, gap = ceiling(time))]
      q <- data.table(ep3 = if (o == 1) as.integer(dte$ep3[ix]) else 0L, gap = dte$gap[ix]); q <- merge(cbind(q, i = seq_along(ix)), bh, by = c('ep3', 'gap'), all.x = TRUE); setorder(q, i)
      lp <- predict(cm, newdata = dte[ix], type = 'lp', reference = 'zero'); h <- ifelse(is.na(q$d), 0, q$d) * exp(lp)
      H[ix, j + 1] <- h
    }
    tot <- rowSums(H); P <- H / pmax(tot, 1e-12) * (1 - exp(-tot)); so <- D$s_prev[te] + 1; P[cbind(seq_len(nrow(P)), so)] <- exp(-tot); leaves <- NA
  }
  fwrite(data.table(firm = D$firm[te], year = D$year[te], s_prev = D$s_prev[te], y = D$y[te], p0 = P[, 1], p1 = P[, 2], p2 = P[, 3], leaves = leaves), out)
  cat(k, meth, leaves, round(as.numeric(difftime(Sys.time(), t0, units = 'secs'))), '\n')
  rm(list = intersect(ls(), c('m', 'dd', 'nd', 'Bn', 'Xt', 'P'))); invisible(gc())
}
