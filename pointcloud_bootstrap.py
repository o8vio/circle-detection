import numpy as np
from sklearn.utils import resample
from gtda import homology
from .featurization import off_diagonal


def fill_dgms(dgms, maxs, maxs_idxs):
    """
    Pad each diagram with zeros or duplicated values
    so that they all have equal shape.
    """

    n_dgms = len(dgms)
    
    filled_dgms = []

    fill_value = dgms[maxs_idxs[1]][-1, 0]

    for dgm in dgms:

        dgm0 = dgm[ dgm[:,2] == 0 ]
        dgm1 = dgm[ dgm[:,2] == 1 ]

        if dgm0.shape[0] < maxs[0]:

            missing = maxs[0] - dgm0.shape[0]
            dgm0 = np.concatenate( ( dgm0, np.zeros((missing, 3)) ) )
        
        if dgm1.shape[0] < maxs[1]:

            missing = maxs[1] - dgm1.shape[0]
            filling = np.c_[np.full((missing, 2), fill_value), np.ones(missing)]
            dgm1 = np.concatenate( ( dgm1, filling ) )
        
        filled_dgms.append(np.concatenate( (dgm0, dgm1) ))
    
    return np.array(filled_dgms).reshape(n_dgms, -1, 3)

            

def bootstrap_dgms(pointclouds, num_subsamples, subsample_size, random_state=None):
    """
    Given a list of pointclouds, resample each one 'R_resamples' times,
    compute and return combined persistence diagrams.
    """
    
    max0, max1 = 0, 0
    max0_idx, max1_idx = 0, 0
    
    combined_dgms = []

    vr_persistence = homology.VietorisRipsPersistence()
    
    for i, pc in enumerate(pointclouds):

        n_points = pc.shape[0]
        
        subsamples = [ resample(pc, replace=False, n_samples=round(subsample_size * n_points), random_state=random_state)
                       for _ in range(num_subsamples) ]
        
        subsamples_dgms = vr_persistence.fit_transform(subsamples)
        
        combined_dgm = off_diagonal(subsamples_dgms.reshape(-1,3))
        
        cdgm0 = combined_dgm[ combined_dgm[:,2] == 0 ]
        cdgm0 = cdgm0[np.argsort(cdgm0[:,1])]
        
        cdgm1 = combined_dgm[ combined_dgm[:,2] == 1 ]
        cdgm1 = cdgm1[np.lexsort((-cdgm1[:,1],-cdgm1[:,0]))]
        
        if cdgm0.shape[0] > max0:

            max0, max0_idx = cdgm0.shape[0], i

        if cdgm1.shape[0] > max1:

            max1, max1_idx = cdgm1.shape[0], i

        combined_dgms.append(np.concatenate((cdgm0, cdgm1)))

    return fill_dgms(combined_dgms, (max0, max1), (max0_idx, max1_idx))