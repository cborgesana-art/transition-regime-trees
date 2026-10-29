"""Minimal example: simulate a three-state panel with three regimes (simulations/sim7.py), grow a transition-regime tree
with the honest criterion and print its splits and the yearly transition probabilities of each leaf."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulations'))
from trtrees import MultiStateTree
from sim7 import simulate, design_matrix

rng = np.random.default_rng(1)
Z, regime, D = simulate('regimes', 500, 'balanced', rng)      # 500 units, 10 attributes, regimes defined by z1 and z2
X1, X2 = design_matrix(D)                                      # leaf designs for the adverse and exit outcomes
g = D.unit.values; y = D.y.values                             # y: 0 healthy, 1 adverse, 2 exit
tree = MultiStateTree(criterion='honest', max_depth=4, min_units_leaf=10, se_rule=1, n_folds=10, random_state=0)
tree.fit(Z[g], X1, X2, y, g)
print('leaves:', tree.n_leaves())
for depth, var, thr in tree.splits():
    print('  ' * depth + f'split on z{var + 1} at {thr:.3f}')
P = tree.predict_proba(Z[g], X1, X2)                           # columns: healthy, adverse, exit at t
leaf = np.array([id(tree._leaf(z)) for z in Z[g]])
for k, l in enumerate(np.unique(leaf)):
    i = leaf == l; s = D.s_prev.values[i]
    print(f'leaf {k + 1}: entry {P[i][s == 0, 1].mean():.3f}, exit from healthy {P[i][s == 0, 2].mean():.3f}, '
          f'recovery {P[i][s == 1, 0].mean():.3f}, exit from adverse {P[i][s == 1, 2].mean():.3f}')
