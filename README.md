# Transition-regime trees

Code for the paper *Transition-regime trees for recurrent states with an absorbing exit, with honest unit-level
splitting* (Ana Borges, Pedro Ribeiro and Mariana Carvalho).

A transition-regime tree partitions units (for example firms) by attributes that are fixed when the unit enters the
data. In each leaf, a Markov model with a healthy state (0), an adverse state (1) and an absorbing exit state (2)
describes the yearly transitions, with multinomial logistic probabilities that depend on lagged covariates, either
directly or as deviations from a global model used as offset. Splits are selected by an honest criterion that leaves
out whole units and repeats the threshold search without them.

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

The comparisons with model-based recursive partitioning and Cox models use R with the packages `partykit`,
`survival` and `data.table`.

## Quick start

```
python example.py
```

simulates 500 units with three regimes, grows a tree with the honest criterion and prints the splits and the transition
probabilities of each leaf. The main class is `trtrees.MultiStateTree`:

```python
from trtrees import MultiStateTree
tree = MultiStateTree(criterion='honest', max_depth=4, min_units_leaf=10, se_rule=1, n_folds=10)
tree.fit(Z, X1, X2, y, groups, offset=None)   # Z: unit attributes (constant within unit), one row per transition
P = tree.predict_proba(Z_new, X1_new, X2_new)  # probabilities of the healthy, adverse and exit states at t
```

`X1` and `X2` are the designs of the linear predictors of the adverse and exit outcomes, with the covariates
interacted with the previous state (see `design_matrix` in `simulations/sim7.py`), `y` is the state at `t` (0, 1, 2) and
`groups` identifies the unit. `criterion` can be `'honest'`, `'aic'`, `'bic'` or `'insample'`; `se_rule` is the constant
`c` of the splitting rule (1 for the one-standard-error rule, `scipy.stats.norm.ppf(1 - 0.05 / q)` for the Bonferroni
version). `offset` is an optional `(n, 2)` array added to the two linear predictors.

## Simulations

```
cd simulations
python sim7.py gen 50 200 2          # data sets: 50 replicates with regimes, 200 without, 2 parallel jobs
python sim7.py fit 2                 # trees fitted in Python
python export.py; Rscript sim7_mob.R # MOB fitted in R
python eval7.py; python tables7.py   # results, table and figure
```

Set `SIM_DIR` to choose where the simulated data sets are written.

## Application

The scripts in `application/` reproduce the analysis of Portuguese manufacturing firms. The SABI data (Bureau van Dijk
and Informa D&B) are licensed and cannot be shared, so these scripts document the analysis and need the user's own
extraction (`SABI_DIR` points to the data and `OUT_DIR` to the results). The main steps are `sabi3_design.py` (data),
`sabi3_full_off.py` (tree on all firms), `sabi3_cv.py`, `sabi3_cv_off.py` and `sabi3_r.R` (five-fold comparison),
`sabi3_eval.py` (evaluation), `sabi3_regimes.py`, `sabi3_tables.py` and `sabi3_fig.py` (regimes), `sabi3_stab.py` and
`sabi3_stab2.py` (stability), `sabi3_temporal.py` (temporal validation) and `sabi3_censor.py` (definitions of exit).

## Licence

MIT.
