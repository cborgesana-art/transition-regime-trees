import numpy as np, pandas as pd, glob, os, sys
for f in sorted(glob.glob(os.environ.get('SIM_DIR', 'simdata') + '/' + (sys.argv[1] if len(sys.argv) > 1 else '') + '*.npz')):
    o = f.replace('.npz', '_tr.csv')
    if os.path.exists(o): continue
    a = np.load(f); Q = a['Z'].shape[1]
    D = pd.DataFrame(a['D'], columns=['unit', 's_prev', 'x1', 'x2', 'y']); D = pd.concat([D, pd.DataFrame(a['Z'][D.unit.astype(int)], columns=[f'z{j+1}' for j in range(Q)])], axis=1)
    Dn = pd.DataFrame(a['Dn'], columns=['unit', 's_prev', 'x1', 'x2', 'y']); Dn = pd.concat([Dn, pd.DataFrame(a['Zn'][Dn.unit.astype(int)], columns=[f'z{j+1}' for j in range(Q)])], axis=1)
    D.to_csv(o, index=False); Dn.to_csv(f.replace('.npz', '_te.csv'), index=False)
