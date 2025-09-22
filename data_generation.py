import numpy as np
from .utils import sample_unit_vector



def generate_circles(num_circles, world_dim=3, r_min=0.25, r_max=0.3):
    
    centers = np.random.rand(num_circles, world_dim)

    radii = (r_max - r_min) * np.random.rand(num_circles) + r_min

    bases = [ np.linalg.qr(np.random.randn(world_dim, 2))[0] 
              for _ in range(num_circles) ]

    return centers, radii, bases




def generate_sample(num_circles, avg_points=200, std_points=10, world_dim=3, 
                    r_min=0.25, r_max=0.3, eps=0.05, sigma=0.05):

    num_points = round(np.random.normal(avg_points, std_points))

    if num_circles == 0:
        
        num_signal_points = 0
        
    else:

        num_signal_points = round((1 - sigma) * num_points)
    
    num_noise_points = num_points - num_signal_points
    
    centers, radii, bases = generate_circles(num_circles, world_dim, r_min, r_max)
    
    points = []

    for i, (center, radius, basis) in enumerate(zip(centers, radii, bases)):

        num_circle_points = num_signal_points // num_circles + 1 if i < num_signal_points % num_circles else num_signal_points // num_circles
        
        for _ in range(num_circle_points):

            angle = 2 * np.pi *np.random.rand()
            
            if world_dim == 2:
                
                points.append( center + radius * np.cos(angle) + radius * np.sin(angle) 
                               + eps * radius * sample_unit_vector(world_dim) )

            else:
                
                points.append(center + radius * np.cos(angle) * basis[:,0] + radius * np.sin(angle) * basis[:,1]
                          + eps * radius * sample_unit_vector(world_dim))

    points += [ np.random.rand(world_dim) for _ in range(num_noise_points) ]

    points = np.array(points)
    
    return centers, radii, bases, points



def generate_dataset(**params):

    sample_params = params.copy()
    
    seed = sample_params.pop('seed')

    if seed is not None:
        np.random.seed(seed)

    num_samples = sample_params.pop('samples_per_class') * 10

    dataset = []
    
    for i in range(num_samples):

        num_circles = i % 10
        
        centers, radii, bases, points = generate_sample(num_circles, **sample_params)

        dataset.append((points, num_circles, centers, radii, bases))
    
    return dataset