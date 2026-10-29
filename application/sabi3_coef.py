import os
import pandas as pd, numpy as np
C = pd.read_csv('out/coef3_off.csv', index_col=0); L = pd.read_csv('out/regimes3_off_labelled.csv').set_index('regime').label
B1 = ['int', 'roa', 'finl', 'log_liq', 'ebm', 'log_ta', 'log_age', 'zm', 'zm2', 'p_2011_13', 'p_2020', 'log_gap']; B2 = ['int', 'roa', 'finl', 'log_liq', 'log_ta', 'log_age', 'zm', 'log_gap']
names = [f'h1_{c}' for c in B1] + [f's1_{c}' for c in B1] + [f'h2_{c}' for c in B2] + [f's2_{c}' for c in B2]
C.columns = names; C['label'] = L.reindex(C.index).values; C = C.sort_values('label', key=lambda s: s.str[1:].astype(int))
sel = ['finl', 'roa', 'log_gap']; blocks = [('h1', 'Healthy to stress'), ('h2', 'Healthy to exit'), ('s1', 'Stress to stress'), ('s2', 'Stress to exit')]
rows = [f"{r.label} & " + ' & '.join(f"{r[f'{b}_{c}']:.2f}" for b, _ in blocks for c in sel) + r"\\" for _, r in C.iterrows()]
head = ' & '.join(r'\multicolumn{3}{c}{' + t + '}' for _, t in blocks)
sub = ' & '.join(['FINL & ROA & log dur.'] * 4)
t = r"""\section{Coefficients of the regimes}\label{a:coef}
\begin{table}[!htbp]\centering\scriptsize
\caption{Selected coefficients of the leaf models of the thirteen regimes. They are deviations from the global spline model used as offset, so a coefficient of zero means that the regime follows the global relation, and they are on the log-odds scale relative to the healthy state at $t$ for transitions from the healthy state, and relative to recovery for transitions from stress. FINL is total liabilities over total assets and ROA the return on assets, both at $t-1$, and log dur.\ the logarithm of the number of consecutive years already spent in the current state. The full sets of coefficients, which also include liquidity, the EBITDA margin, size, age, the Zmijewski score and its square and indicators of the transitions into the years 2011 to 2013 (sovereign debt crisis) and 2020 (first year of the COVID-19 pandemic), are available with the code.}\label{t:coef}
\resizebox{\textwidth}{!}{\begin{tabular}{l""" + 'r' * 12 + r"""}\toprule
 & """ + head + r"""\\
 & """ + sub + r"""\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table}\n"
open(os.environ.get('TEX_DIR', 'tex') + '/tab_coef.tex', 'w').write(t)
print(C[[f'{b}_{c}' for b, _ in blocks for c in sel]].round(2).to_string())
