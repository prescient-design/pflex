import numpy as np
from collections import defaultdict

#ds = np.load("/gstore/scratch/u/nerlis/pmhc/oct_runs/ft_relaxed_sweep/test_results_all_v3_sweep.npy", allow_pickle=True)
ds = np.load("/gstore/scratch/u/nerlis/pmhc/v7_082324/set0/topmean_set0.npy", allow_pickle=True)

print (ds)

vars = {}
#12
#71

vars['distances_mean'] = np.take((ds.item().get('distances_mean')).reshape((143, 15, 34)),[12, 17], axis=0)
vars['distances_std'] = np.take((ds.item().get('distances_std')).reshape((143, 15, 34)),[12, 17], axis=0)

# read the inferred phi dihedrals
vars['phi_mean'] = np.take((ds.item().get('phi_mean')).reshape((143, 15, 1)),[12, 17], axis=0)
vars['phi_std'] = np.take((ds.item().get('phi_std')).reshape((143, 15, 1)),[12, 17], axis=0)

# read the inferred psi dihedrals
vars['psi_mean'] = np.take((ds.item().get('psi_mean')).reshape((143, 15, 1)),[12, 17], axis=0)
vars['psi_std'] = np.take((ds.item().get('psi_std')).reshape((143, 15, 1)),[12, 17], axis=0)

new_ds = np.array(vars)

np.save("/gstore/scratch/u/nerlis/pmhc/pmhc_flex/example/long/test.npy", new_ds, allow_pickle=True)

