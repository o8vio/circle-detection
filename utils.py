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