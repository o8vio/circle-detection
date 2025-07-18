import numpy as np, matplotlib.pyplot as plt


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


def plot_scores(x_values, train_scores, val_scores, title, shape=None, **kwargs):

    if shape == None:

        t_means = train_scores.mean(axis=1)
        t_stds = train_scores.std(axis=1)
        v_means = val_scores.mean(axis=1)
        v_stds = val_scores.std(axis=1)
        
        plt.figure(figsize=(6,6))
        plt.title(title)
        
        plt.plot(x_values, t_means, '-o', color='tab:blue', label='training score')
        plt.plot(x_values, v_means, '-o', color='tab:green', label='validation score')
        plt.fill_between(x_values, t_means - t_stds, t_means + t_stds,
                         alpha=0.1, color='tab:blue')
        plt.fill_between(x_values, v_means - v_stds, v_means + v_stds,
                         alpha=0.1, color='tab:green')
        plt.grid()
        plt.tight_layout()
        plt.legend(loc='lower right')
        filename = title.replace(' ', '_')+'.pdf'
        plt.savefig(filename)
    
    else:

        rows, cols = shape
        
        fig, axs = plt.subplots(rows, cols, figsize=(10, 4*rows))

        for i, scores in enumerate(zip(train_scores, val_scores)):

            i_row, i_col = i // cols, i % cols
            
            axs[i_row, i_col].set_title(title[i])

            t_means = scores[0].mean(axis=1)
            t_stds = scores[0].std(axis=1)
            v_means = scores[1].mean(axis=1)
            v_stds = scores[1].std(axis=1)

            axs[i_row, i_col].plot(x_values, t_means, '-o', color='tab:blue', label='training score')
            axs[i_row, i_col].fill_between(x_values, t_means - t_stds, t_means + t_stds, 
                                           alpha=0.1, color='tab:blue')
            axs[i_row, i_col].plot(x_values, v_means, '-o', color='tab:green', label='validation score')
            axs[i_row, i_col].fill_between(x_values, v_means - v_stds, v_means + v_stds, 
                                           alpha=0.1, color='tab:green')

            axs[i_row, i_col].legend(loc='lower right', fontsize=8)
            axs[i_row, i_col].set_xlabel(kwargs['x_label'])
            axs[i_row, i_col].set_ylabel(kwargs['y_label'])
        
        plt.savefig('many_scores.pdf')