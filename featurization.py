import numpy as np



""" bar filtering """

def off_diagonal(dgm):

    return dgm[ dgm[:,1] - dgm[:,0] > 0 ]


def bars_in_range_interval(dgm, min, max=1):

    dgm = off_diagonal(dgm)
    max_persistence = np.max( dgm[:,1] - dgm[:,0] ) if dgm.shape[0] > 0 else 0

    interval = max_persistence * min, max_persistence * max

    return dgm[ (dgm[:,1] - dgm[:,0] >= interval[0]) & (dgm[:,1] - dgm[:,0] <= interval[1]) ]


def bars_in_interval(dgm, center, radius, strict):

    dgm = off_diagonal(dgm)
    max_death = np.max( dgm[:,1] ) if dgm.shape[0] > 0 else 0
    
    interval = max_death * (center - radius), max_death * (center + radius)

    return dgm[ (dgm[:,0] >= interval[0]) & (dgm[:,1] <= interval[1]) ] if strict else dgm[ (dgm[:,0] < interval[1]) & (dgm[:,1] > interval[0]) ]


def jth_max_bar(dgm, j):

    return dgm[ np.flip( np.argsort( dgm[:,1] - dgm[:,0] ) )[j-1] ] if dgm.shape[0] >= j else np.full((3,), 0)





""" diagram scalar features """

def count_bars(dgm):

    return dgm.shape[0]


def sum_bars(dgm):

    return np.sum( dgm[:,1] - dgm[:,0] )


def mean_persistence(dgm):

    return sum_bars(dgm) / count_bars(dgm) if dgm.shape[0] > 0 else 0


def std_persistence(dgm):

    return np.std( dgm[:,1] - dgm[:,0] ) if dgm.shape[0] > 0 else 0