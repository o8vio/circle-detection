import numpy as np
from .experiments import pc_subsample_experiment



rf_params = { 'n_estimators' : 20, 
              'max_features' : 'sqrt', 
              'max_depth' : 8, 
              'min_samples_leaf' : 8 }


tp_params = { 'samples_per_class' : 20, 
              'avg_points' : 500, 
              'std_points' : 25, 
              'r_min' : 0.25, 
              'r_max' : 0.3, 
              'eps' : 0.05 }


bootstrap_params = { 'num_subsamples' : 30,
                     'subsample_size' : 0.25,
                     'random_state' : 8 }


sigma_range = np.linspace(0.05,0.4, 8)


feature_dicts = [ {'feature_name':'betti_curves', 'params':{'n_bins':80}}, 
                  [{'feature_name':'persistence_entropy'} ,
                   {'feature_name': 'amplitude', 'params': {'metric':'bottleneck'} },
                   {'feature_name':'count_bars', 'params':{'dim':0, 'low':0, 'high':0.6}}, 
                   {'feature_name':'count_bars', 'params':{'dim':1, 'low':0, 'high':0.6}}, 
                   {'feature_name':'sum_bars', 'params':{'dim':0, 'low':0, 'high':0.6}}, 
                   {'feature_name':'sum_bars', 'params':{'dim':1, 'low':0, 'high':0.6}}] ]


num_seeds = 90


pc_subsample_experiment(rf_params=rf_params,
                        tp_params=tp_params,
                        bootstrap_params=bootstrap_params,
                        sigma_range=sigma_range,
                        feature_dicts=feature_dicts, 
                        num_seeds=num_seeds,
                        filename='topocircles/experiments/results/bootstrap_results.npy')