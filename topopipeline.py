import numpy as np
from gtda import homology, diagrams
from .data_generation import generate_dataset
from .featurization import *
from .utils import plot_pointcloud
from .pointcloud_bootstrap import bootstrap_dgms


class topopipeline:

    def __init__(self, data, data_params, bootstrap_params):

        self.data_params = data_params.copy()
        self.bootstrap_params = bootstrap_params.copy()
        
        self.num_samples = len(data)
        
        self.pointclouds = [ pc for pc, _, _, _, _ in data ]
        self.labels = [ lb for _, lb, _, _, _ in data ]
        self.circles = [ (cs, rs, bs) for _, _, cs, rs, bs in data ]
        
        if self.bootstrap_params['num_subsamples'] is not None:

            self.diagrams = bootstrap_dgms(self.pointclouds, **bootstrap_params)

        else:
            
            self.diagrams = homology.VietorisRipsPersistence().fit_transform(self.pointclouds)
    
    
    @staticmethod
    def random(samples_per_class=20, avg_points=200, std_points=10, 
               world_dim=3, r_min=0.25, r_max=0.3, eps=0.05, sigma=0.05, seed=None,
               num_subsamples=None, subsample_size=None, random_state=None):
        
        data_params = {'seed': seed,
                       'samples_per_class' : samples_per_class,
                       'avg_points' : avg_points,
                       'std_points' : std_points,
                       'world_dim' : world_dim,
                       'r_min' : r_min,
                       'r_max' : r_max,
                       'eps' : eps,
                       'sigma' : sigma }
        
        bootstrap_params = { 'num_subsamples' : num_subsamples,
                             'subsample_size' : subsample_size,
                             'random_state' : random_state }
        
        data = generate_dataset(**data_params)
        
        return topopipeline(data, data_params, bootstrap_params)
    
    
    
    def plot(self, idx):

        plot_pointcloud(self.pointclouds[idx], self.labels[idx], self.data_params['sigma'])
    
    
    def get_labels(self):
        
        return np.array(self.labels)



    """ bar filtering """

    def _get_subdiagram(self, dgm, dim, **kwargs):

        dgm = dgm[ dgm[:, 2] == dim ]
        
        if 'pmin' in kwargs:
            return bars_in_range_interval(dgm, **kwargs)
        
        elif 'low' in kwargs:
            return bars_in_interval(dgm, **kwargs)
        
        else:
            return off_diagonal(dgm)
    
    
    def _get_jth_bar(self, dgm, j):

        return jth_max_bar(dgm, j)
    
    
    
    """ vectorizations """

    def betti_curves(self, **kwargs):

        return diagrams.BettiCurve(**kwargs).fit_transform(self.diagrams).reshape(self.num_samples, -1)
    
    
    def persistence_images(self, **kwargs):

        return diagrams.PersistenceImage(**kwargs).fit_transform(self.diagrams).reshape(self.num_samples, -1)
    

    def persistence_landscapes(self, **kwargs):

        return diagrams.PersistenceLandscape(**kwargs).fit_transform(self.diagrams).reshape(self.num_samples, -1)
    

    
    
    """ features """
    
    def persistence_entropy(self, **kwargs):

        return diagrams.PersistenceEntropy(**kwargs).fit_transform(self.diagrams).reshape(self.num_samples, -1)
    
    
    def amplitude(self, **kwargs):

        return diagrams.Amplitude(**kwargs).fit_transform(self.diagrams).reshape(self.num_samples, -1)
    
    
    
    def count_bars(self, dim, **kwargs):

        return np.array([ count_bars(self._get_subdiagram(dgm, dim, **kwargs)) for dgm in self.diagrams ]).reshape(self.num_samples, -1)
    

    def sum_bars(self, dim, **kwargs):

        return np.array([ sum_bars(self._get_subdiagram(dgm, dim, **kwargs)) for dgm in self.diagrams ]).reshape(self.num_samples, -1)
    

    def mean_persistence(self, dim, **kwargs):

        return np.array([ mean_persistence(self._get_subdiagram(dgm, dim, **kwargs)) for dgm in self.diagrams ]).reshape(self.num_samples, -1)
    

    def std_persistence(self, dim, **kwargs):

        return np.array([ std_persistence(self._get_subdiagram(dgm, dim, **kwargs)) for dgm in self.diagrams ]).reshape(self.num_samples, -1)
    
    
    
    def jth_max_persistence(self, dim, j, **kwargs):

        return np.array([ sum_bars(self._get_jth_bar(self._get_subdiagram(dgm, dim, **kwargs), j)) for dgm in self.diagrams ]).reshape(self.num_samples, -1)
    
    
    def jth_max_birth(self, dim, j, **kwargs):

        return np.array([ self._get_jth_bar(self._get_subdiagram(dgm, dim, **kwargs), j)[0,0] for dgm in self.diagrams ]).reshape(self.num_samples, -1)


    def jth_max_death(self, dim, j, **kwargs):

        return np.array([ self._get_jth_bar(self._get_subdiagram(dgm, dim, **kwargs), j)[0,1] for dgm in self.diagrams ]).reshape(self.num_samples, -1)