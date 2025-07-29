import numpy as np, matplotlib.pyplot as plt
import json


def sample_unit_vector(dim):

    v = np.random.randn(dim)

    return v / np.linalg.norm(v)


def plot_pointcloud(pointcloud, num_circles, sigma):
    
    total_points, world_dim = pointcloud.shape
    
    if num_circles == 0:
        
        signal_points = 0
        
    else:
        
        signal_points = int((1-sigma)*total_points)
        points_per_circle = [ signal_points // num_circles + 1 if N < signal_points % num_circles
                              else signal_points // num_circles for N in range(num_circles) ]

    if world_dim == 2:

        plt.figure(figsize=(6,6))
        plt.axis('equal')

        start = 0

        for count in points_per_circle:

            end = start + count
            plt.scatter(pointcloud[start:end, 0], pointcloud[start:end, 1],
                        s=10, alpha=1)
            start = end
        
        plt.scatter(pointcloud[start:, 0],
                   pointcloud[start:, 1], s=10, alpha=1)
        
        plt.show()
        

    elif world_dim == 3:

        ax = plt.axes(projection='3d')
        #ax.set_box_aspect((np.ptp(pointcloud[:,0]), np.ptp(pointcloud[:,1]), np.ptp(pointcloud[:,2])))
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        ax.set_box_aspect([1,1,1])
        
        start = 0

        for count in points_per_circle:

            end = start + count
            ax.scatter(pointcloud[start:end, 0], pointcloud[start:end, 1], pointcloud[start:end, 2],
                        s=10, alpha=1)
            start = end
        
        ax.scatter(pointcloud[start:, 0], pointcloud[start:, 1], pointcloud[start:, 2],
                    s=10, alpha=1)
        
        plt.show()
    
    else:

        print('world_dim > 3')


def multiclass_brier_score_loss(y_true, y_probs):

    N, K = y_probs.shape
    
    loss = 0

    for i in range(N):

        yi = np.zeros(K)
        yi[y_true[i]] = 1

        loss += np.sum((yi - y_probs[i])**2)
    
    return loss/N


def plot_scores(param_range, results, **kwargs):

  n_features = len(results)

  rows = n_features // 2 + 1 if n_features % 2 == 1 else n_features // 2

  fig, axs = plt.subplots(rows, 2, figsize=(5*rows, 10))

  titles = kwargs['titles']

  if kwargs['param_name'] == 'world dim':

    dim_values = param_range.copy()
    param_range = np.arange(len(dim_values))

  for i, feature_key in enumerate(results):

    i_row, i_col = i // 2, i % 2

    axs[i_row, i_col].set_title(titles[i])

    t_means, t_stds = results[feature_key]['train_score'].mean(axis=1), results[feature_key]['train_score'].std(axis=1)
    v_means, v_stds = results[feature_key]['test_score'].mean(axis=1), results[feature_key]['test_score'].std(axis=1)

    axs[i_row, i_col].plot(param_range, t_means, '-o', color='tab:blue', label='training score')
    axs[i_row, i_col].plot(param_range, v_means, '-o', color='tab:green', label='validation score')

    if kwargs['fill']:

      axs[i_row, i_col].fill_between(param_range, t_means - t_stds, t_means + t_stds, alpha=0.1, color='tab:blue')
      axs[i_row, i_col].fill_between(param_range, v_means - v_stds, v_means + v_stds, alpha=0.1, color='tab:green')

    if kwargs['param_name'] == 'world dim':

      axs[i_row, i_col].set_xticks(param_range)
      axs[i_row, i_col].set_xticklabels(dim_values)

    axs[i_row, i_col].legend(loc='lower right', fontsize=8)
    axs[i_row, i_col].set_xlabel(kwargs['param_name'])
    axs[i_row, i_col].set_ylabel(kwargs['y_label'])

  if kwargs['show']:
     
     plt.show()


def save_results(results, filename):
   
   for feature_key in results:
      for score_key in results[feature_key]:
         results[feature_key][score_key] = results[feature_key][score_key].tolist()
   
   with open(filename, 'w') as f:
    json.dump(results, f)


def load_results(file):
   
    with open(file, 'r') as file:
      
      results = json.load(file)
   
    for feature_key in results:
      for score_key in results[feature_key]:
         results[feature_key][score_key] = np.array(results[feature_key][score_key])
    
    return results