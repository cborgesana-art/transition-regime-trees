suppressMessages({library(partykit); library(data.table)})
softmax <- function(E1, E2) { m <- pmax(0, E1, E2); A <- cbind(exp(-m), exp(E1 - m), exp(E2 - m)); A / rowSums(A) }
P1 <- 6
mnfit <- function(y, x, start = NULL, weights = NULL, offset = NULL, ..., estfun = FALSE, object = FALSE) {
  P <- ncol(x); beta <- if (is.null(start)) rep(0, P) else start; x1 <- x[, 1:P1, drop = FALSE]; x2 <- x[, (P1 + 1):P, drop = FALSE]
  obj <- function(b) { pr <- softmax(drop(x1 %*% b[1:P1]), drop(x2 %*% b[(P1 + 1):P])); -sum(log(pmax(pr[cbind(seq_along(y), y + 1)], 1e-300))) + 5e-4 * sum(b^2) }
  for (it in 1:25) {
    pr <- softmax(drop(x1 %*% beta[1:P1]), drop(x2 %*% beta[(P1 + 1):P])); r1 <- pr[, 2] - (y == 1); r2 <- pr[, 3] - (y == 2)
    g <- c(crossprod(x1, r1), crossprod(x2, r2)) + 1e-3 * beta
    H11 <- crossprod(x1 * (pr[, 2] * (1 - pr[, 2])), x1); H22 <- crossprod(x2 * (pr[, 3] * (1 - pr[, 3])), x2); H12 <- -crossprod(x1 * (pr[, 2] * pr[, 3]), x2)
    st <- solve(rbind(cbind(H11, H12), cbind(t(H12), H22)) + diag(1e-3, P), g); f0 <- obj(beta); lam <- 1
    while (lam > 1e-4 && obj(beta - lam * st) > f0 + 1e-12) lam <- lam / 2
    beta <- beta - lam * st; if (max(abs(lam * st)) < 1e-8) break
  }
  pr <- softmax(drop(x1 %*% beta[1:P1]), drop(x2 %*% beta[(P1 + 1):P])); names(beta) <- colnames(x)
  ef <- if (estfun) cbind(x1 * ((y == 1) - pr[, 2]), x2 * ((y == 2) - pr[, 3])) else NULL
  list(coefficients = beta, objfun = -sum(log(pmax(pr[cbind(seq_along(y), y + 1)], 1e-300))), estfun = ef, object = NULL)
}
des <- function(d) { a <- as.numeric(d$s_prev == 0); b <- 1 - a; B <- cbind(1, d$x1, d$x2); X <- cbind(B * a, B * b); X <- cbind(X, X); colnames(X) <- paste0('w', 1:12); X }
fs <- sort(Sys.glob(file.path(Sys.getenv('SIM_DIR', 'simdata'), '*_tr.csv')))
for (f in fs) {
  out <- sub('_tr.csv', '_mob.csv', f); if (file.exists(out)) next
  d <- fread(f); dn <- fread(sub('_tr.csv', '_te.csv', f)); X <- des(d); ZC <- paste0('z', 1:10)
  dd <- data.frame(y = d$y, X, d[, ..ZC], unit = d$unit)
  fml <- as.formula(paste('y ~', paste(colnames(X), collapse = '+'), '- 1 |', paste(ZC, collapse = '+')))
  m <- tryCatch(mob(fml, data = dd, fit = mnfit, cluster = unit, control = mob_control(maxdepth = 5, minsize = 80, alpha = 0.05, bonferroni = TRUE, ytype = 'vector', xtype = 'matrix')), error = function(e) NULL)
  if (is.null(m)) next
  res <- list()
  for (s0 in 0:1) {
    dn2 <- copy(dn); dn2$s_prev <- s0; Xn <- des(dn2); nid <- predict(m, newdata = data.frame(y = dn2$y, Xn, dn2[, ..ZC]), type = 'node')
    cf <- coef(m); if (is.null(dim(cf))) cf <- matrix(cf, nrow = 1, dimnames = list(as.character(nodeids(m, terminal = TRUE)), names(cf)))
    B <- cf[as.character(nid), , drop = FALSE]; P <- softmax(rowSums(Xn[, 1:6] * B[, 1:6]), rowSums(Xn[, 7:12] * B[, 7:12]))
    res[[s0 + 1]] <- P
  }
  zn <- unique(dn[, c('unit', ZC), with = FALSE]); leafn <- predict(m, newdata = data.frame(y = 0, des(cbind(zn, s_prev = 0, x1 = 0, x2 = 0)), zn[, ..ZC]), type = 'node')
  rl <- partykit:::.list.rules.party(m); sv <- unique(unlist(regmatches(rl, gregexpr('z[0-9]+', rl))))
  fwrite(data.table(a0 = res[[1]][, 1], a1 = res[[1]][, 2], a2 = res[[1]][, 3], b0 = res[[2]][, 1], b1 = res[[2]][, 2], b2 = res[[2]][, 3],
                    leaves = length(nodeids(m, terminal = TRUE)), nvars = paste(sv, collapse = ';')), out)
  fwrite(data.table(unit = zn$unit, leaf = leafn), sub('_tr.csv', '_mobleaf.csv', f))
}
cat('MOBDONE\n')
