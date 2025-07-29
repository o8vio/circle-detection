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

eps_values = np.linspace(0,1,21)
n_eps = len(eps_values)

feature_keys = ['mix', 'betti', 'landscape', 'image']
score_keys = ['train_score', 'test_score']

n_seeds = 80

exp1_results = { feature+'_results' : { score_key : np.zeros(shape=(n_eps, n_seeds)) for score_key in score_keys }
                 for feature in feature_keys }


for i, eps in enumerate(eps_values):
  
  for j in range(n_seeds):

    tp = topopipeline.random(seed=j, samples_per_class=20, avg_points=250, std_points=15, 
                             r_min=0.25, r_max=0.3, eps=eps, sigma=0)

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

      exp1_results[feature_key+'_results']['test_score'][i, j] = model.score(X_test, y_test)
      exp1_results[feature_key+'_results']['train_score'][i, j] = model.score(X_train, y_train)

      print(f'current param: {i+1}/{n_eps} , current seed: {j+1}/{n_seeds}', end='\r', flush=True)

plot_scores(param_range=eps_values, results=exp1_results,
            titles=[key+' results' for key in feature_keys],
            fill=True, y_label='accuracy', param_name='eps', show=False)

plt.savefig('./experiments/results/exp1_results.pdf')

save_results(exp1_results, './experiments/results/exp1_results.json')