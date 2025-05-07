
import numpy as np
from numba import jit
from numba import njit

@jit(nopython=True)
def calculate_cosine_difference(PHI, PSI, OMEGA, N):
    
    D = np.zeros((N, N), dtype=np.float64)
    indices = list(range(0, len(PHI)))

    for idx1 in indices:
        for idx2 in indices[(idx1+1):]:
            if idx1 == idx2:
                continue

            phiarr1 = PHI[idx1,:]
            phiarr2 = PHI[idx2,:]

            psiarr1 = PSI[idx1,:]
            psiarr2 = PSI[idx2,:]

            omegaarr1 = OMEGA[idx1,:]
            omegaarr2 = OMEGA[idx2,:]
            #JIT version should be a bit faster given that the overhead of going from numpy operations 
            #may not be worthwhile
            phi_diff = np.max(2*(np.subtract(1, np.cos(np.subtract(phiarr1, phiarr2)))))
            psi_diff = np.max(2*(np.subtract(1, np.cos(np.subtract(psiarr1, psiarr2)))))
            omega_diff = np.max(2*(np.subtract(1, np.cos(np.subtract(omegaarr1, omegaarr2)))))

            dih_dist_i_j = max(phi_diff, psi_diff, omega_diff)

            D[idx1, idx2] = dih_dist_i_j
            D[idx2, idx1] = dih_dist_i_j

    return D
