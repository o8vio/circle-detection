import numpy as np
from utils import sample_unit_vector



def generate_pointcloud(num_circles, avg_points=200, std_points=10, world_dim=3, r_min=0.1, r_max=0.4, eps=0.1, sigma=0.1, seed=None):

    if seed is not None:
        np.random.seed(seed)

    total_points = int(np.random.normal(avg_points, std_points))

    if num_circles == 0:
        
        signal_points = 0
        
    else:

        signal_points = int((1 - sigma) * total_points)
        points_per_circle = [ signal_points // num_circles + 1 if N < signal_points % num_circles
                              else signal_points // num_circles for N in range(num_circles) ]
    
    noise_points = total_points - signal_points
    
    points = []

    for N in range(num_circles):

        center = np.random.rand(world_dim)
        radius = (r_max - r_min) * np.random.rand() + r_min
        basis = np.linalg.qr(np.random.randn(world_dim, 2))[0]

        for _ in range(points_per_circle[N]):

            angle = 2 * np.pi *np.random.rand()
            points.append(center + radius * np.cos(angle) * basis[:,0] + radius * np.sin(angle) * basis[:,1]
                          + eps * radius * sample_unit_vector(world_dim))

    points += [ np.random.rand(world_dim) for _ in range(noise_points) ]
    
    return np.array(points)



def generate_dataset(**params):

    seed = params.pop('seed')

    num_samples = params.pop('samples_per_class') * 10

    return [ (generate_pointcloud(i % 10, **params, seed=seed*i), i % 10) for i in range(num_samples) ]