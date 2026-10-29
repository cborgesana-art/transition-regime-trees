import os
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = pd.read_csv(os.environ.get('SIM_OUT', '.') + '/sim7_raw.csv'); r = R[R.scen == 'regimes']; n = R[R.scen == 'null']; q = R[R.scen == 'rare']
names = {'honest, 1-SE': 'Honest, 1-SE rule', 'honest, Bonferroni': 'Honest, Bonferroni', 'BIC': 'BIC', 'AIC': 'AIC', 'MOB': 'MOB', 'oracle partition': 'Oracle partition'}
rows = []
for d in ['balanced', 'unbalanced']:
    if d == 'unbalanced': rows.append(r'\midrule')
    first = True
    for meth, lab in names.items():
        for i, m in enumerate([100, 250, 500]):
            x = r[(r.design == d) & (r.method == meth) & (r.m == m)]; se = lambda c: x[c].std() / np.sqrt(len(x))
            ns = '--' if meth == 'oracle partition' else f"{(n[(n.design == d) & (n.method == meth) & (n.m == m)].leaves > 1).mean():.3f}"
            rows.append(f"{d if first else ''} & {lab if i == 0 else ''} & {m} & {x.rmse_entry.mean():.3f} & {x.rmse_recovery.mean():.3f} & {x.rmse_exit1.mean():.3f} & {x.logloss.mean():.4f} ({se('logloss'):.4f}) & {x.ari.mean():.2f} ({se('ari'):.2f}) & {x.leaves.mean():.1f} & {x.spurious.mean():.2f} & {ns}\\\\")
            first = False
cap = r"""\caption{Simulation with three regimes (50 replicates per cell) and with no regimes (200 replicates per cell). Errors are computed on 1000 new firms. RMSE, root mean squared error of the yearly probability of entry (0 to 1), recovery (1 to 0) and exit from the adverse state (1 to 2); Log-loss, mean negative log-likelihood of the transitions of the new firms; ARI, adjusted Rand index between the leaf of a new firm and its true regime; Monte Carlo standard errors of the log-loss and the ARI in parentheses; Leaves, mean number of leaves; Spur., mean number of uninformative attributes used; Null splits, proportion of replicates of the scenario without regimes in which the tree splits at least once.}\label{t:sim}"""
t = "\\begin{table}[!htbp]\\centering\\footnotesize\n" + cap + "\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrrrrrlrrr}\\toprule\nDesign & Method & $m$ & RMSE 0$\\to$1 & RMSE 1$\\to$0 & RMSE 1$\\to$2 & Log-loss & ARI & Leaves & Spur. & Null splits\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table}\n"
open(os.environ.get('TEX_DIR', 'tex') + '/tab_sim.tex', 'w').write(t)
cols = {'honest, 1-SE': '#33507a', 'BIC': '#c0703a', 'MOB': '#5a9a5a', 'AIC': '#999999', 'honest, Bonferroni': '#8fa8cf'}
fig, ax = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
for j, d in enumerate(['balanced', 'unbalanced']):
    for meth, c in cols.items():
        x = r[(r.design == d) & (r.method == meth)].groupby('m'); mm = x.ari.mean(); s = x.ari.std() / np.sqrt(x.size())
        ax[0, j].errorbar(mm.index, mm, yerr=1.96 * s, marker='o', color=c, label=names[meth], capsize=3)
        if meth == 'AIC': continue
        o = r[(r.design == d) & (r.method == 'oracle partition')].set_index(['m', 'rep']).logloss; y = r[(r.design == d) & (r.method == meth)].set_index(['m', 'rep']).logloss
        dd = (y - o).groupby(level=0); mm = dd.mean(); s = dd.std() / np.sqrt(dd.size())
        ax[1, j].errorbar(mm.index, mm, yerr=1.96 * s, marker='o', color=c, capsize=3)
    ax[0, j].set_title(f'{d.capitalize()} design', fontsize=10); ax[0, j].set_ylabel('ARI'); ax[1, j].set_ylabel('Log-loss minus oracle'); ax[1, j].axhline(0, color='k', lw=.6); ax[1, j].set_xlabel('Number of training firms')
    for a in ax[:, j]: a.set_xticks([100, 250, 500]); a.grid(alpha=.3)
ax[0, 0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.environ.get('TEX_DIR', 'tex') + '/fig_sim.png', dpi=200)

rows = []
for d in ['balanced', 'unbalanced']:
    if d == 'unbalanced': rows.append(r'\midrule')
    first = True
    for meth, lab in names.items():
        for i, m in enumerate([250, 500, 1000]):
            x = q[(q.design == d) & (q.method == meth) & (q.m == m)]
            if not len(x): continue
            se = lambda c: x[c].std() / np.sqrt(len(x))
            rows.append(f"{d if first else ''} & {lab if i == 0 else ''} & {m} & {x.rmse_exit1.mean():.3f} & {x.logloss.mean():.4f} ({se('logloss'):.4f}) & {x.ari.mean():.2f} ({se('ari'):.2f}) & {x.leaves.mean():.1f} & {x.spurious.mean():.2f}\\\\")
            first = False
ex = q[q.method == 'global'].groupby('m').apply(lambda d: 0)
cap = r"""\caption{Simulation with three regimes and rare exits (exit intercepts lowered by one; 50 replicates per cell, with on average 19 to 24, 41 to 48 and 80 to 96 exits per training set for 250, 500 and 1000 firms). Columns as in Table~\ref{t:sim}.}\label{t:rare}"""
t = "\\begin{table}[!htbp]\\centering\\footnotesize\n" + cap + "\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrrllrr}\\toprule\nDesign & Method & $m$ & RMSE 1$\\to$2 & Log-loss & ARI & Leaves & Spur.\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table}\n"
open(os.environ.get('TEX_DIR', 'tex') + '/tab_rare.tex', 'w').write(t)
