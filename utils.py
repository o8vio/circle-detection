import numpy as np, matplotlib.pyplot as plt
from itertools import count
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


def plot_results(param_name, param_range, results, **kwargs):

  n_features = results.shape[2]

  n_rows, n_columns = kwargs.get('n_rows', int(np.ceil(n_features / 2))), kwargs.get('n_columns', 2)

  fig, axs = plt.subplots(n_rows, n_columns, constrained_layout=True, figsize=kwargs.get('figsize', (5 * n_columns, 5 * n_rows)))
  axs = np.atleast_1d(axs).flatten()

  titles = kwargs['titles']

  if param_name == 'world_dim':

    dim_values = param_range.copy()
    param_range_ = np.arange(len(dim_values))
  
  else:
     
     param_range_ = param_range.copy()

  for i in range(n_features):

    ax = axs[i]

    ax.set_title(titles[i])
    
    ax.set_ylim(bottom=0, top=1)

    t_means = results[:,:,i,0].mean(axis=1)
    t_stds = results[:,:,i,0].std(axis=1)
    v_means = results[:,:,i,1].mean(axis=1)
    v_stds = results[:,:,i,1].std(axis=1)

    ax.plot(param_range_, t_means, '-o', color='tab:blue', label='training score')
    ax.plot(param_range_, v_means, '-o', color='tab:green', label='validation score')

    if kwargs.get('fill', True):

      ax.fill_between(param_range_, t_means - t_stds, t_means + t_stds, alpha=0.1, color='tab:blue')
      ax.fill_between(param_range_, v_means - v_stds, v_means + v_stds, alpha=0.1, color='tab:green')

    ax.set_xticks(param_range_)
    
    if param_name == 'world_dim':

      ax.set_xticklabels(dim_values)

    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlabel(param_name)
    ax.set_ylabel(kwargs.get('y_label', 'accuracy'))
  
  for j in range(n_features, len(axs)):
     
     fig.delaxes(axs[j])
  
  if kwargs.get('save', True):
     
     fig.savefig(kwargs.get('figname', './figure.pdf'))
  
  if kwargs.get('show', True):
     
     plt.show()

  else:
     
     plt.close(fig)


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