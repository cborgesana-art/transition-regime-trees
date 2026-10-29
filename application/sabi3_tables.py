import os
import pandas as pd, numpy as np, re
R = pd.read_csv(os.environ.get('OUT_DIR', 'out') + '/regimes3_off.csv').sort_values('exit10_h').reset_index(drop=True)
def fmt(var, lo, hi):
    tr = {'z_log_ta0': ('A$_0$', lambda v: f'{np.exp(v)/1000:.1f}'), 'z_finl0': ('FINL$_0$', lambda v: f'{v:.2f}'), 'z_roa0': ('ROA$_0$', lambda v: f'{100*v:.1f}\\%'),
          'z_log_emp0': ('Emp$_0$', lambda v: f'{np.expm1(v):.0f}'), 'z_first_year': ('Entry', lambda v: f'{int(np.floor(v))}'), 'z_founded': ('Founded', lambda v: f'{int(np.floor(v))}'),
          'z_state0': ('state$_0$', None), 'z_cae2': ('CAE div.', lambda v: f'{int(np.floor(v))}'), 'z_cae': ('CAE', lambda v: f'{int(np.floor(v))}'),
          'z_tech': ('tech', lambda v: f'{int(np.floor(v))}'), 'z_log_liq0': ('log LIQ$_0$', lambda v: f'{v:.2f}')}
    nm, f = tr[var]
    if var == 'z_state0': return 'stressed$_0$' if lo is not None else 'healthy$_0$'
    if var == 'z_cae2':
        a = 10 if lo is None else int(np.floor(lo)) + 1; b = 33 if hi is None else int(np.floor(hi)); return f'CAE {a} to {b}'
    if var == 'z_tech':
        return 'low tech' if hi is not None else 'medium or high tech'
    if var == 'z_first_year':
        if lo is not None and hi is not None: return f'entry {int(lo)+1} to {int(hi)}'
        return f'entry $\\le${int(hi)}' if hi is not None else f'entry $\\ge${int(lo)+1}'
    if lo is not None and hi is not None: return f'{f(lo)} $<$ {nm} $\\le$ {f(hi)}'
    return f'{nm} $\\le$ {f(hi)}' if hi is not None else f'{nm} $>$ {f(lo)}'
def rule(s):
    b = {}
    for part in s.split(' & '):
        m = re.match(r'(\S+) (<=|>) (\S+)', part); v, op, th = m.group(1), m.group(2), float(m.group(3))
        lo, hi = b.get(v, (None, None))
        if op == '>': lo = th if lo is None else max(lo, th)
        else: hi = th if hi is None else min(hi, th)
        b[v] = (lo, hi)
    order = ['z_log_ta0', 'z_finl0', 'z_roa0', 'z_log_emp0', 'z_first_year', 'z_founded', 'z_state0', 'z_cae2', 'z_cae', 'z_tech', 'z_log_liq0']
    return ', '.join(fmt(v, *b[v]) for v in order if v in b)
R['rule'] = R.regime.map(rule); R['label'] = [f'R{i+1}' for i in range(len(R))]
R.to_csv(os.environ.get('OUT_DIR', 'out') + '/regimes3_off_labelled.csv', index=False)
ci = lambda r, c, k=2: f"{r[c]:.{k}f} ({r[c+'_lo']:.{k}f}, {r[c+'_hi']:.{k}f})"
rows = [f"{r.label} & {r.rule} & {r.firms:,} & {r.p01:.3f} & {r.p02:.3f} & {r.p10:.3f} & {r.p12:.3f} & {ci(r,'exit_share')} & {r.stress_spell:.1f} & {ci(r,'exit10_h')}\\\\" for _, r in R.iterrows()]
t = r"""\begin{table}[!htbp]\centering\footnotesize
\caption{Regimes of the three-state transition-regime tree fitted to all firms, with leaves that model deviations from a global spline model, ordered by the probability of exit within ten years for a firm that starts healthy. A$_0$ is total assets in thousands of euros, FINL$_0$ total liabilities over total assets, ROA$_0$ the return on assets and Emp$_0$ the number of employees, all in the first year in which the firm is observed, CAE the five-digit code of the activity and CAE 10 to 12 a range of divisions, and low tech the divisions of low technological intensity (10 to 18, 31 and 32). The four transition columns are mean predicted yearly probabilities of entry into stress (0 to 1), exit from the healthy state (0 to 2), recovery (1 to 0) and exit from stress (1 to 2). Exit share is the probability that a stress episode ends in exit rather than recovery, the stress spell is its expected length in years and Exit 10y the probability of exit within ten years for a firm that starts healthy, all average-probability summaries (Section~\ref{s:summ}). Intervals are 95\% bootstrap intervals (200 resamples of firms within each regime, conditional on the partition).}\label{t:regimes}
\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrlrl}\toprule
& Rule & Firms & 0$\to$1 & 0$\to$2 & 1$\to$0 & 1$\to$2 & Exit share & Stress spell & Exit 10y\\\midrule
""" + "\n".join(rows).replace('&,', '&').replace(',', '{,}', 0) + "\n\\bottomrule\\end{tabular}}\\end{table}\n"
t = re.sub(r'(\d),(\d{3})', r'\1{,}\2', t)
open(os.environ.get('TEX_DIR', 'tex') + '/tab_regimes.tex', 'w').write(t)
print(R[['label', 'rule', 'firms', 'p01', 'p02', 'p10', 'p12', 'obs_p01', 'obs_p02', 'obs_p10', 'obs_p12', 'exit_share', 'stress_spell', 'exit10_h', 'exit10_s']].round(3).to_string())
