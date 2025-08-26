import numpy as np


def deterministic_guess(topopipeline, low, high):

    num_instances = topopipeline.num_samples
    guesses = np.full(num_instances, -1)

    for i in range(num_instances):
        
        a0 = topopipeline.count_bars(dim=0, low=0, high=high)
        a1 = topopipeline.count_bars(dim=1, low=low, high=high)

        guess = round((a0 + a1)/2)

        guesses[i] = np.min(guess, 9)   
    
    return guesses