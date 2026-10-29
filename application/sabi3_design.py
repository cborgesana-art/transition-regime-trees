"""Three-state design: healthy (0), stress (1), exit (2, absorbing). A firm whose current status in SABI is dissolution,
liquidation, insolvency, closed or extinct is taken to exit in the year after its last year with a known state, provided
that year is not after 2024. Other firms (active, inactive, merged) are censored after their last known state.
Covariates are measured at t-1, attributes as in the two-state design."""
import os
import pandas as pd, numpy as np, sys
EXIT = {'Dissolução', 'Liquidação', 'Insolvência', 'Encerrada', 'Extinção'}
def build():
    d = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/wide.pkl'); L = pd.read_pickle(os.environ.get('SABI_DIR', 'data') + '/long.pkl')
    def w(x, lo=.01, hi=.99): a, b = x.quantile([lo, hi]); return x.clip(a, b)
    first = L[L.state.notna()].groupby('firm').first(); last = L[L.state.notna()].groupby('firm').year.max()
    cae = d.CAE.astype(int); cae2 = cae // 1000; tech = {21: 4, 26: 4, 20: 3, 27: 3, 28: 3, 29: 3, 30: 3, 19: 2, 22: 2, 23: 2, 24: 2, 25: 2, 33: 2}
    found = pd.to_numeric(d['Ano_Constituição'], errors='coerce')
    Z = pd.DataFrame({'z_cae': cae, 'z_cae2': cae2, 'z_tech': cae2.map(lambda c: tech.get(c, 1)), 'z_founded': found}).loc[first.index]
    Z['z_first_year'] = first.year; Z['z_log_ta0'] = np.log(first.TA.clip(lower=1)); Z['z_finl0'] = first.FINL; Z['z_roa0'] = first.ROA
    Z['z_log_liq0'] = np.log(first.LIQ.clip(lower=1e-3)); Z['z_log_emp0'] = np.log1p(first.EMP.fillna(0)); Z['z_state0'] = first.state
    exiting = d['Situação_Actual'].isin(EXIT)
    L = L.assign(last=L.firm.map(last), ex=L.firm.map(exiting))
    obs = L.state_l1.notna() & L.state.notna()
    exitrow = L.state_l1.notna() & L.state.isna() & (L.year == L['last'] + 1) & L.ex.fillna(False).astype(bool)
    R = L[obs | exitrow].copy(); R.loc[exitrow[obs | exitrow].values, 'state'] = 2
    R = R[R.year > R.firm.map(first.year)]
    X = pd.DataFrame({'roa': w(R.ROA_l1), 'finl': w(R.FINL_l1), 'log_liq': w(np.log(R.LIQ_l1.clip(lower=1e-3))), 'ebm': w(R.EBM_l1.replace([np.inf, -np.inf], np.nan)),
                      'log_ta': np.log(R.TA_l1.clip(lower=1)), 'log_age': np.log((R.year - R.firm.map(found)).clip(lower=1)), 'zm': w(R.zm_l1),
                      'p_2011_13': R.year.between(2011, 2013).astype(int), 'p_2020': (R.year == 2020).astype(int)}, index=R.index)
    X['zm2'] = X.zm ** 2
    D = pd.concat([R[['firm', 'year', 'state', 'state_l1']].rename(columns={'state': 'y', 'state_l1': 's_prev'}), X], axis=1).join(Z, on='firm').dropna()
    D['status'] = D.firm.map(d['Situação_Actual'])
    # time already spent in the current state (consecutive years ending at t-1) and episode number of stress
    D = D.sort_values(['firm', 'year']).reset_index(drop=True)
    f = D.firm.values; sp = D.s_prev.values; yr = D.year.values; gap = np.ones(len(D), int); ep = np.zeros(len(D), int)
    for i in range(len(D)):
        same = i > 0 and f[i] == f[i - 1] and yr[i] == yr[i - 1] + 1
        if same and sp[i] == sp[i - 1]: gap[i] = gap[i - 1] + 1
        if i > 0 and f[i] == f[i - 1]: ep[i] = ep[i - 1] + int(sp[i] == 1 and not (same and sp[i - 1] == 1))
        else: ep[i] = int(sp[i] == 1)
    D['gap'] = gap; D['episode'] = ep; D['log_gap'] = np.log(gap)
    return D
XC = ['roa', 'finl', 'log_liq', 'ebm', 'log_ta', 'log_age', 'zm', 'zm2', 'p_2011_13', 'p_2020', 'log_gap']
XE = ['roa', 'finl', 'log_liq', 'log_ta', 'log_age', 'zm', 'log_gap']           # smaller set for the exit equation
def design(D):
    """Blocks: [stress equation | exit equation], each interacted with the previous state."""
    a = (D.s_prev.values == 0).astype(float)[:, None]; b = 1 - a
    B1 = np.column_stack([np.ones(len(D)), D[XC].values]); B2 = np.column_stack([np.ones(len(D)), D[XE].values])
    return np.column_stack([B1 * a, B1 * b]), np.column_stack([B2 * a, B2 * b])
if __name__ == '__main__':
    D = build(); D.to_pickle(os.environ.get('SABI_DIR', 'data') + '/trans3.pkl'); D.drop(columns=['status']).to_csv(os.environ.get('SABI_DIR', 'data') + '/trans3.csv', index=False)
    print(D.shape, D.firm.nunique()); print(pd.crosstab(D.s_prev, D.y, normalize='index').round(4)); print(pd.crosstab(D.s_prev, D.y))
    print(D[D.y == 2].status.value_counts()); print(D[D.y == 2].year.value_counts().sort_index().to_dict())
