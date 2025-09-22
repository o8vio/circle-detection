import numpy as np, matplotlib.pyplot as plt, inspect
from itertools import product


def filter_kwargs(func, kwargs):
    
    sig = inspect.signature(func)
    accepted = set(sig.parameters)

    return {k: v for k, v in kwargs.items() if k in accepted}


def sample_unit_vector(dim):

    v = np.random.randn(dim)

    return v / np.linalg.norm(v)


def plot_pointcloud(pointcloud, num_circles, sigma):
    

    num_points, world_dim = pointcloud.shape
    
    if num_circles == 0:
        
        num_signal_points = 0
        
    else:
        
        num_signal_points = round((1-sigma)*num_points)

    if world_dim == 2:

        plt.figure(figsize=(6,6))
        plt.axis('equal')

        start = 0

        for i in range(num_circles):
           
          num_circle_points = num_signal_points // num_circles + 1 if i < num_signal_points % num_circles else num_signal_points // num_circles
           
          end = start + num_circle_points

          plt.scatter(pointcloud[start:end, 0], pointcloud[start:end, 1], s=10, alpha=1)

          start = end
        
        plt.scatter(pointcloud[start:, 0], pointcloud[start:, 1], s=10, alpha=1)
        
        plt.show()        

    elif world_dim == 3:

        ax = plt.axes(projection='3d')
        #ax.set_box_aspect((np.ptp(pointcloud[:,0]), np.ptp(pointcloud[:,1]), np.ptp(pointcloud[:,2])))
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        ax.set_box_aspect([1,1,1])
        
        start = 0

        for i in range(num_circles):
           
          num_circle_points = num_signal_points // num_circles + 1 if i < num_signal_points % num_circles else num_signal_points // num_circles
           
          end = start + num_circle_points

          ax.scatter(pointcloud[start:end, 0], pointcloud[start:end, 1], pointcloud[start:end, 2], s=10, alpha=1)

          start = end
        
        ax.scatter(pointcloud[start:, 0], pointcloud[start:, 1], pointcloud[start:, 2], s=10, alpha=1)
        
        plt.show()
    
    else:

        print('cannot plot if world_dim > 3')



def multiclass_brier_score_loss(y_true, y_probs, flip=True):

    N, K = y_probs.shape
    
    loss = 0

    for i in range(N):

        yi = np.zeros(K)
        yi[y_true[i]] = 1

        loss += np.sum((yi - y_probs[i])**2)
    
    return loss/N if not flip else 1 - loss/N



def plot_experiment_results(param_name, param_range, results, 
                            feat_keys=['Betti curves', 'Features mix', 'Persistence landscapes', 'Persistence images'], 
                            metrics=['accuracy', 'Brier score loss'], 
                            score_types=['train', 'test'],
                            bootstrap=['sin bootstrap', 'con bootstrap'], 
                            split_by=None, **kwargs):


    axis_map = { 'feature' : (-3, feat_keys), 
                 'metric' : (-2, metrics), 
                 'score_type' : (-1, score_types),
                 **({'boot' : (0, bootstrap)} if results.ndim ==  6 else {}) }

    colors = iter(kwargs['colors']) if 'colors' in kwargs else iter(['tab:blue', 'tab:cyan', 'tab:red', 'tab:pink', 'tab:green', 
                                                               'lightgreen', 'olive', 'yellowgreen', 'turquoise', 'aquamarine',
                                                               'darkviolet', 'violet', 'dimgray', 'darkgray', 'darkkhaki', 'khaki'])
    
    if split_by == None:

        num_features, num_metrics, num_scoretype = results.shape[-3:]
        
        fig = plt.figure(**filter_kwargs(plt.figure, kwargs))

        if 'title' in kwargs:

            plt.title(kwargs['title'], **filter_kwargs(plt.title, kwargs))

        for k, l, m in product(range(num_features), range(num_metrics), range(num_scoretype)):

            values = results[..., k,l,m]

            means = values.mean(axis=1)
            stds = values.std(axis=1)

            plt.plot(param_range, means, '-o', label=string_j+' '+string_k, color=next(colors))
            plt.fill_between(param_range, means - stds, means + stds, color=next(colors), alpha=0.1)
        
        plt.xlabel(param_name)
        plt.ylabel('score')
        plt.legend(loc='lower right')
        plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    
    else:

        split_axis, split_strings = axis_map.pop(split_by)
        
        n_subplots = results.shape[split_axis]
        
        rows, cols = (n_subplots + 1) // 2, 1 if n_subplots == 1 else 2

        fig, axs = plt.subplots(rows, cols, **filter_kwargs(plt.subplots, kwargs), 
                                            **filter_kwargs(plt.figure, kwargs))

        axs = axs.flatten()
        
        inner_cats = list(axis_map.values())
        axis_idx_label_lists = [ [(axis, (idx, label)) for idx, label in enumerate(labels)] 
                                for axis, labels in inner_cats ]

        if 'title' in kwargs:
            
            plt.title(kwargs['title'], **filter_kwargs(plt.title, kwargs))

        
        for i in range(n_subplots):

            ax = axs[i]

            ax.set_title(split_strings[i])
            
            for curve in product( *axis_idx_label_lists ):

                slicer = [slice(None)] * results.ndim
                slicer[split_axis] = i
                
                if results.ndim == 5:
                    (axis1, (j, string_j)), (axis2, (k, string_k)) = curve
                    slicer[axis1] = j
                    slicer[axis2] = k
                    
                else:
                    (axis1, (j, string_j)), (axis2, (k, string_k)), (axis3, (l, string_l)) = curve
                    slicer[axis1] = j
                    slicer[axis2] = k
                    slicer[axis3] = l
                    
                values = results[tuple(slicer)]
                
                means = values.mean(axis=1)
                stds = values.std(axis=1)

                curve_color = next(colors)
                ax.plot(param_range, means, '-o', label=string_j+' '+string_k+' '+string_l, color=curve_color)
                ax.fill_between(param_range, means - stds, means + stds, color=curve_color, alpha=0.1)

            
            ax.set_xlabel(param_name)
            ax.set_ylabel('score')
            ax.legend(loc='lower right')
            ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    if 'filename' in kwargs:

        fig.savefig(kwargs['filename'], dpi=fig.dpi)
    
    else:

        plt.show()