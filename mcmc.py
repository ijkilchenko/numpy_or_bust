from collections import defaultdict
from numpy.random import RandomState
import numpy as np

num_points = 4

def _f():
    return [89, 77, 84, 30]

def f(x):
    """This is the function f(x) which is proportional to some P(x). 

    In this example, consider 4 points in a circle as our space. We 
    do not know the actual probablity distribution, but we do know 
    some relative probabilities. """

    return _f()[x]

def metropolis(f=f):
    x = np.random.randint(num_points)

    def next_x_given_x(x):
        """This gives the next point located uniformly around x. 
        This is the jumping distribition. """
        half_width = 1
        next_x = np.random.randint(x-half_width, x+half_width+1) # half open [)
        # loop around since our points are in a circle 
        return next_x % num_points # note: this works with negative numbers too

    bins_of_samples = defaultdict(lambda : 0)
    num_samples = 0

    num_iter = 100000
    for i in range(num_iter):
        next_x = next_x_given_x(x)
        acceptance_ratio = f(next_x)/f(x) # = P(x)/P(next_x)
        if acceptance_ratio >= 1 or np.random.uniform() < acceptance_ratio:
            # accept next_x 
            x = next_x
            bins_of_samples[x] += 1 # count this sample
            num_samples += 1
        else:
            # reject next_x
            pass
    stationary_distribution = [bins_of_samples[p]/num_samples for p in sorted(bins_of_samples)]
    return stationary_distribution

print('target distribution:')
denominator_P = sum(_f())
print([float('%.2f'%round(x/denominator_P, 2)) for x in _f()])

print('stationary distribution:')
print([round(x, 2) for x in metropolis()])

