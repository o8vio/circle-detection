import numpy as np
from sklearn.utils import resample
from gtda import homology
from featurization import off_diagonal


def fill_dgms(dgms, maxs, max_idx):

    n_dgms = len(dgms)
    
    filled_dgms = []

    fill_value = dgms[max_idx][-1, 0]

    for dgm in dgms:

        dgm0 = dgm[ dgm[:,2] == 0 ]
        dgm1 = dgm[ dgm[:,2] == 1 ]

        if dgm0.shape[0] < maxs[0]:

            missing = maxs[0] - dgm0.shape[0]
            dgm0 = np.concatenate( ( dgm0, np.zeros((missing, 3)) ) )
        
        if dgm1.shape[0] < maxs[1]:

            missing = maxs[1] - dgm1.shape[0]
            filling = np.c_[np.full((missing, 2), fill_value), np.full(missing, 1)]
            dgm1 = np.concatenate( ( dgm1, filling ) )
        
        filled_dgms.append(np.concatenate( (dgm0, dgm1) ))
    
    return np.array(filled_dgms).reshape(n_dgms, -1, 3)

            

def bootstrap_dgms(pointclouds, **params):

    R_samples, resample_size = params['R_resamples'], params['resample_size']
    rs = params['random_state']

    max0, max1 = 0, 0
    max1_idx = 0
    
    combined_dgms = []
    
    for i, pointcloud in enumerate(pointclouds):

        N_points = pointcloud.shape[0]
        resamples = [ resample(pointcloud, replace=False, n_samples=int(resample_size*N_points), random_state=rs)
                      for _ in range(R_samples) ]
        
        resamples_dgms = homology.VietorisRipsPersistence().fit_transform(resamples)
        
        combined_dgm = off_diagonal(resamples_dgms.reshape(-1,3))
        
        cdgm0 = combined_dgm[ combined_dgm[:,2] == 0 ]
        cdgm0 = cdgm0[np.argsort(cdgm0[:,1])]
        
        cdgm1 = combined_dgm[ combined_dgm[:,2] == 1 ]
        cdgm1 = cdgm1[np.lexsort((-cdgm1[:,1],-cdgm1[:,0]))]
        
        combined_dgms.append(np.concatenate((cdgm0, cdgm1)))

        if cdgm0.shape[0] > max0:

            max0 = cdgm0.shape[0]

        if cdgm1.shape[0] > max1:

            max1 = cdgm1.shape[0]
            max1_idx = i
    
    return fill_dgms(combined_dgms, (max0, max1), max1_idx)