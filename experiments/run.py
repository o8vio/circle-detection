import numpy as np
from .experiments import param_range_experiment


rf_params = { 'n_estimators' : 20, 
              'max_features' : 'sqrt', 
              'max_depth' : 8, 
              'min_samples_leaf' : 8 }


tp_params = { 'samples_per_class' : 20, 
              'avg_points' : 150, 
              'std_points' : 15, 
              'r_min' : 0.25, 
              'r_max' : 0.3, 
              'eps' : 0.05, 
              'sigma' : 0.05 }


num_seeds = 90


feature_dicts = [ {'feature_name':'betti_curves', 'params':{'n_bins':80}}, 
                  {'feature_name':'persistence_landscapes', 'params':{'n_layers' : 10, 'n_bins':80}},
                  {'feature_name':'persistence_images', 'params':{'n_bins':80}},
                  [{'feature_name':'persistence_entropy'} ,
                   {'feature_name': 'amplitude', 'params': {'metric':'bottleneck'} },
                   {'feature_name':'count_bars', 'params':{'dim':0, 'low':0, 'high':0.6}}, 
                   {'feature_name':'count_bars', 'params':{'dim':1, 'low':0, 'high':0.6}}, 
                   {'feature_name':'sum_bars', 'params':{'dim':0, 'low':0, 'high':0.6}}, 
                   {'feature_name':'sum_bars', 'params':{'dim':1, 'low':0, 'high':0.6}}] ]


params = [ 'avg_points', 'eps', 'sigma', 'world_dim', 'r_range']

param_ranges = [ np.linspace(50,500, 10, dtype=int),
                 np.linspace(0, 1, 21),
                 np.linspace(0, 0.5, 11),
                 np.linspace(2,30,29, dtype=int),
                 np.linspace(0, 0.5, 11) ]


for (i, param_name), param_range in zip(enumerate(params), param_ranges):

    param_range_experiment(rf_params=rf_params,
                           tp_params=tp_params, 
                           param_name=param_name, 
                           param_range=param_range,
                           feature_dicts=feature_dicts, 
                           num_seeds=num_seeds,
                           filename='topocircles/experiments/results/exp'+str(i)+'_results.npy')