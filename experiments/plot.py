import numpy as np
from ..utils import plot_experiment_results


params = [ 'avg_points', 'eps', 'sigma', 'world_dim', 'r_range']

param_ranges = [ np.linspace(50,500, 10, dtype=int),
                 np.linspace(0, 1, 21),
                 np.linspace(0, 0.5, 11),
                 np.linspace(2,30,29, dtype=int),
                 np.linspace(0, 0.5, 11) ]

feature_keys = [ 'betti curve', 'features mix' ]

for i, (param, param_range) in enumerate(zip(params, param_ranges)):

    results = np.load('topocircles/experiments/results/exp'+str(i)+'_results.npy')
    
    filename = 'topocircles/experiments/results/exp'+str(i)+'_plott.pdf'

    plot_experiment_results(param, param_range, results, split_by='feature',
                            figsize=(12,12), filename=filename)