import sys
sys.path.append('C:\\Users\\Octavio\\Desktop\\tesis\\topocircles\\')


from topopipeline import *
from utils import plot_scores, save_results
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np, matplotlib.pyplot as plt



rf_params = { 'n_estimators' : 25,
              'max_features' : 'sqrt',
              'max_depth' : 9,
              'min_samples_leaf' : 8 }

r_range_values = np.linspace(0,0.6,13)
n_ranges = len(r_range_values)

feature_keys = ['mix', 'betti', 'landscape', 'image']
score_keys = ['train_score', 'test_score']

n_seeds = 80

exp4_results = { feature+'_results' : { score_key : np.zeros(shape=(n_ranges, n_seeds)) for score_key in score_keys }
                 for feature in feature_keys }


for i, r in enumerate(r_range_values):
  
  for j in range(n_seeds):

    tp = topopipeline.random(seed=j, samples_per_class=20, avg_points=250, std_points=15, 
                             world_dim=3, r_min=0.3 - r/2, r_max=0.3 + r/2, eps=0.05, sigma=0.05)

    X_mix = np.c_[tp.persistence_entropy(),
                  tp.amplitude(metric='bottleneck'),
                  tp.count_bars(dim=0, min=0.25),
                  tp.sum_bars(dim=0, min=0.25),
                  tp.count_bars(dim=1, min=0.25),
                  tp.sum_bars(dim=1, min=0.25),
                  tp.jth_max_persistence(dim=1, j=1),
                  tp.jth_max_birth(dim=1, j=1),
                  tp.jth_max_death(dim=1, j=1)]

    X_betti = tp.betti_curves()

    X_landscape = tp.persistence_landscapes(n_layers=10)

    X_images = tp.persistence_images()

    y = tp.get_labels()

    features = dict(zip(feature_keys, [X_mix, X_betti, X_landscape, X_images]))

    for feature_key in features:

      model = RandomForestClassifier(random_state=j, **rf_params)

      X_train, X_test, y_train, y_test = train_test_split(features[feature_key], y, stratify=y, test_size=0.25, random_state=j)

      model.fit(X_train, y_train)

      exp4_results[feature_key+'_results']['test_score'][i, j] = model.score(X_test, y_test)
      exp4_results[feature_key+'_results']['train_score'][i, j] = model.score(X_train, y_train)

      print(f'current param: {i+1}/{n_ranges} , current seed: {j+1}/{n_seeds}', end='\r', flush=True)

plot_scores(param_range=r_range_values, results=exp4_results,
            titles=[key+' results' for key in feature_keys],
            fill=True, y_label='accuracy', param_name='r_max - r_min', show=False)

plt.savefig('./experiments/results/exp4_results.pdf')

save_results(exp4_results, './experiments/results/exp4_results.json')