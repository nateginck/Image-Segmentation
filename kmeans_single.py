import numpy as np

# function to compute kmeans_single
def kmeans_single(X, K, iters):
    m, n = X.shape

    # find min and max
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    # declare random means
    means = np.random.uniform(mins, maxs, (K, n))

    # create column to determine cluster
    ids = np.zeros(m)

    for iteration in range(iters):
        # calculate distance for every point and center
        distances = np.zeros((m, K))
        for i in range(m):
            for j in range(K):
                distances[i, j] = np.sqrt(np.sum((X[i] - means[j]) ** 2))

        # classify on closest cluster
        ids = np.argmin(distances, axis=1)

        # recalculate each mean cluster
        for k in range(K):
            if np.any(ids == k):
                means[k] = np.mean(X[ids == k], axis=0)

    # Calculated sum squared distances
    ssd = sum((np.linalg.norm(X[ids == k] - means[k], axis=1) ** 2).sum() for k in range(K))

    # return final values
    return ids, means, ssd

