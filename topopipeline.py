import numpy as np
from gtda import homology, diagrams
from sklearn.utils import resample
from data_generation import *
from featurization import *
from utils import plot_pointcloud
from pointcloud_bootstrap import bootstrap_dgms


class topopipeline:

    def __init__(self, data, data_params, bootstrap, bootstrap_params):

        self.data_params = data_params
        self.bootstrap_params = bootstrap_params
        self.num_samples = len(data)
        self.pointclouds = [ pointcloud for pointcloud, _ in data ]
        self.labels = [ label for _, label in data ]
        if bootstrap:

            self.diagrams = bootstrap_dgms(self.pointclouds, **bootstrap_params)

        else:
            
            self.diagrams = homology.VietorisRipsPersistence().fit_transform(self.pointclouds)
    
    
    @staticmethod
    def random(seed=8, samples_per_class=20, avg_points=200, std_points=10, 
               world_dim=3, r_min=0.1, r_max=0.3, eps=0.1, sigma=0.1,
               bootstrap=False, R_resamples=None, resample_size=None, random_state=None):
        
        data_params = {'seed': seed,
                       'samples_per_class' : samples_per_class,
                       'avg_points' : avg_points,
                       'std_points' : std_points,
                       'world_dim' : world_dim,
                       'r_min' : r_min,
                       'r_max' : r_max,
                       'eps' : eps,
                       'sigma' : sigma }
        
        bootstrap_params = { 'R_resamples' : R_resamples,
                             'resample_size' : resample_size,
                             'random_state' : random_state }
        
        data = generate_dataset(**data_params)
        
        return topopipeline(data, data_params, bootstrap, bootstrap_params)
    
    @staticmethod
    def from_file(filename):

        content = np.load(filename, allow_pickle=True)
        
        data, data_params, bootstrap_params = list(content[0]), content[1], content[2]
        
        return topopipeline(data, data_params, bootstrap_params)
    
    
    def save(self, filename):

        data = list( zip(self.pointclouds, self.labels) )
        data_params = self.data_params
        bootstrap_params = self.bootstrap_params

        np.save(filename, np.array( (data, data_params, bootstrap_params), dtype=object), 
                allow_pickle=True)
    
    
    def plot(self, idx):

        plot_pointcloud(self.pointclouds[idx], idx % 10, self.data_params['sigma'])
    
    
    def get_labels(self):
        
        return np.array(self.labels)



    """ bar filtering """

    def _get_subdiagram(self, dgm, dim, **kwargs):

        dgm = dgm[ dgm[:, 2] == dim ]
        
        if 'min' in kwargs:
            return bars_in_range_interval(dgm, kwargs['min'], kwargs['max']) if 'max' in kwargs else bars_in_range_interval(dgm, kwargs['min'])
        
        elif 'center' in kwargs:
            return bars_in_interval(dgm, kwargs['center'], kwargs['radius'], kwargs['strict'])
        
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
