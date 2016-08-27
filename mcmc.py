import numpy as np
from collections import defaultdict


def f(x):
    """This is the function f(x) which is proportional to some P(x). 

    In this example, consider 5 points in a circle as our space. We
    do not know the actual probability distribution, but we do know
    some relative probabilities. """

    return _f()[x]


def next_x_given_x(x, width=2):
    """This gives the next point located normally (rounded to nearest integer)
    around x. This is the jumping distribution. """
    num_points = len(_f())

    next_x = round(np.random.normal()*width)

    # Loop around since our points are in a circle.
    # Note: this works with negative numbers too.
    return next_x % num_points


def metropolis(fun=f, num_iter=10000):
    """fun is the function f(x) which is proportional to some P(x) which we
    are trying to approximate. """

    num_points = len(_f())
    x = np.random.randint(num_points)  # Starting point.

    # We count how many times we observe each point in the Markov sequence.
    bins_of_samples = defaultdict(lambda: 0)

    # Could be less than num_iter because we reject some samples.
    for i in range(num_iter):
        next_x = next_x_given_x(x)
        acceptance_ratio = fun(next_x)/fun(x)  # = P(next_x)/P(x).
        if acceptance_ratio >= 1 or np.random.uniform() < acceptance_ratio:
            # Accept next_x.
            x = next_x
        else:
            # Reject next_x.
            pass
        bins_of_samples[x] += 1  # Count this sample.
    stationary_distribution = [bins_of_samples[p]/num_iter
                               for p in bins_of_samples]
    return stationary_distribution

if __name__ == '__main__':
    for D in [[89, 77, 84, 1, 30],
              [89, 77, 84, 1, 300]]:

        def _f():
            return D

        print('Target distribution: ')
        denominator_P = sum(_f())
        print([float('%.2f' % round(x/denominator_P, 2)) for x in _f()])

        print('Stationary distribution: ')
        print([round(x, 2) for x in metropolis()])

        print()
