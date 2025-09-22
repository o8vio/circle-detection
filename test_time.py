import time
import numpy as np
from .topopipeline import topopipeline

"""avg_points_range= np.linspace(50,500,10,dtype=int)

for ap in avg_points_range:

    inicio = time.time()

    tpb = topopipeline.random(seed=1, avg_points=ap, sigma=0.4)

    fin = time.time()
    
    print(f"Tiempo de ejecución {ap} pts: {fin - inicio:.6f} segundos", end='\n')"""

inicio = time.time()

tp = topopipeline.random(seed=1, avg_points=1000, sigma=0.4)

fin = time.time()

print(f"Tiempo de ejecución {1000} pts: {fin - inicio:.6f} segundos", end='\n')