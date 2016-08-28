import numpy as np
import pandas as pd
from collections import defaultdict
from bokeh.charts import Scatter, show


def dist(x, y):
    """Euclidean distance between x and y. """
    return np.linalg.norm(x-y)


def init_centroids(X, k):
    centroids = [X[np.random.randint(0, len(X)-1)]]  # First random point.
    for i in range(1, k):  # For each remaining centroid.
        D = assign_to_clusters(X, centroids)
        # Choose new centroid point (from X) according to weights in D.
        D = [d[0] for d in D]
        D_sum = sum(D)
        D = [d/D_sum for d in D]
        centroids.append(X[np.random.choice(range(len(X)), p=D)])
    return centroids


def assign_to_clusters(X, centroids):
    """For each data point, calculate the distance to the closest
    centroid, and put that distance into D. """
    D = []
    for j, x in enumerate(X):
        for m, centroid in enumerate(centroids):
            curr_dist = dist(centroid, x)
            if len(D) - 1 < j:
                D.append((curr_dist, m))
            else:
                if D[-1][0] > curr_dist:
                    D[-1] = (curr_dist, m)
    return D


def recalc_centroids(X, D):
    cluster_to_indices = defaultdict(lambda: [])
    for i, d in enumerate(D):
        cluster_to_indices[d[1]].append(i)

    centroids = []
    for cluster in cluster_to_indices:
        new_cluster = [0]*len(X[0])
        for x in cluster_to_indices[cluster]:
            x = X[x]
            for i, x_i in enumerate(x):
                new_cluster[i] += x_i
        new_cluster = [c/len(cluster_to_indices[cluster]) for c in new_cluster]
        centroids.append(new_cluster)
    return centroids


def kmeanspp(X, k, num_iter=100):
    centroids = init_centroids(X, k)

    for _ in range(num_iter):
        D = assign_to_clusters(X, centroids)
        centroids = recalc_centroids(X, D)

    df = pd.DataFrame(X)
    D = [d[1] for d in D]
    df = pd.concat([df, pd.Series(D)], axis=1)
    df.columns = ['x', 'y', 'cluster']

    p = Scatter(df, x='x', y='y', color='cluster')

    show(p)

if __name__ == '__main__':
    X = [np.array([np.random.uniform(0, 100), np.random.uniform(0, 100)])
         for _ in range(500)]

    kmeanspp(X, k=5)
