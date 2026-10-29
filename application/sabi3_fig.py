import os
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = pd.read_csv(os.environ.get('OUT_DIR', 'out') + '/regimes3_off_labelled.csv')
fig, ax = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True); y = np.arange(len(R))[::-1]
spec = [('p10', 'p12', 'Yearly probability from stress', ['Recovery (1 to 0)', 'Exit (1 to 2)']), ('exit_share', None, 'Share of stress episodes ending in exit', None), ('exit10_h', 'exit10_s', 'Exit within ten years', ['Starting healthy', 'Starting in stress'])]
for a, (c1, c2, title, lab) in zip(ax, spec):
    if c2 is None:
        a.errorbar(R[c1], y, xerr=[R[c1] - R[c1 + '_lo'], R[c1 + '_hi'] - R[c1]], fmt='o', color='#33507a', ms=5, capsize=2)
    else:
        for c, off, col, l in [(c1, .15, '#33507a', lab[0]), (c2, -.15, '#c0703a', lab[1])]:
            lo = R[c + '_lo'] if c + '_lo' in R else R[c]; hi = R[c + '_hi'] if c + '_hi' in R else R[c]
            a.errorbar(R[c], y + off, xerr=[R[c] - lo, hi - R[c]], fmt='o', color=col, ms=5, capsize=2, label=l)
        a.legend(frameon=False, fontsize=8, loc='upper center' if c1 == 'p10' else 'upper right')
    a.set_title(title, fontsize=10); a.grid(axis='x', alpha=.3); a.set_xlim(left=0)
ax[0].set_yticks(y); ax[0].set_yticklabels(R.label)
plt.tight_layout(); plt.savefig(os.environ.get('TEX_DIR', 'tex') + '/fig_regimes.png', dpi=200)
