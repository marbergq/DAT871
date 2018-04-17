import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import multiprocessing as mp
import time


def generateData(n, plot=None):
    X, y = make_blobs(n_samples=n, cluster_std=1.7,
                      shuffle=False, random_state=2122)
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.scatter(X[:, 0], X[:, 1])
        axes.set_xticks([])
        axes.set_yticks([])
        plt.show()
    return X


def nearestCentroid(datum, centroids):
    # norm(a-b) is Euclidean distance, matrix - vector computes difference
    # for all rows of matrix
    dist = np.linalg.norm(centroids - datum, axis=1)
    return np.argmin(dist), np.min(dist)


def nearestCentroid_paritioned_parallel(indicies, data, c, cluster_sizes, variation, centroids):
    for i in indicies:
        cluster, dist = nearestCentroid(data[i], centroids)
        c[i] = cluster
        cluster_sizes[cluster] += 1
        variation[cluster] += dist**2
    return

def map_to_nearest_centroid(indicies, data, centroid, result_arr):
    for i in indicies:
        cluster, dist = nearestCentroid(data[i], centroid)

def kmeans_serial(k, data, nr_iter=100):
    N = len(data)

    # Choose k random data points as centroids
    centroids = data[np.random.choice(
        np.array(range(N)), size=k, replace=False)]
    print "Initial centroids\n", centroids

    N = len(data)

    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)

    print "Iteration\tVariation\tDelta Variation"
    total_variation = 0.0
    for j in range(nr_iter):
        #print "=== Iteration %d ===" % (j+1)
        # Assign data points to nearest centroid
        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)
        for i in range(N):
            cluster, dist = nearestCentroid(data[i], centroids)
            c[i] = cluster
            cluster_sizes[cluster] += 1
            variation[cluster] += dist**2
        delta_variation = -total_variation
        total_variation = sum(variation)
        delta_variation += total_variation
        print "%3d\t\t%f\t%f" % (j, total_variation, delta_variation)

        # Recompute centroids
        centroids = np.zeros((k, 2))
        for i in range(N):
            centroids[c[i]] += data[i]
        centroids = centroids / cluster_sizes.reshape(-1, 1)

        #print "Total variation", total_variation
        #print "Cluster sizes", cluster_sizes
        #print c
        #print centroids

    return total_variation, c

def kmeans_paralell(k, data, nr_iter=100):
    N = len(data)

    # Choose k random data points as centroids
    centroids = data[np.random.choice(
        np.array(range(N)), size=k, replace=False)]
    print "Initial centroids\n", centroids

    N = len(data)
    c_array = mp.Array("i", N)
    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    print "Iteration\tVariation\tDelta Variation"
    total_variation = 0.0
    partitioned = np.array_split(range(N), k)

    for j in range(nr_iter):
        #print "=== Iteration %d ===" % (j+1)

        # Assign data points to nearest centroid
        variation = mp.Array("d", k)
        cluster_sizes = mp.Array("i", k)
        start_time = time.time()
        workers = []

        for z in range(k):
            workers.append(mp.Process(target=nearestCentroid_paritioned_parallel,
                                      args=(partitioned[z], data, c_array, cluster_sizes, variation, centroids)))
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        delta_variation = -total_variation
        total_variation = sum(variation)
        delta_variation += total_variation
        print "%3d\t\t%f\t%f" % (j, total_variation, delta_variation)
        print("--- %s seconds ---" % (time.time() - start_time))

        cluster_sizes_arr = np.frombuffer(
            cluster_sizes.get_obj(), dtype="int32")
        # Recompute centroids
        centroids = np.zeros((k, 2))
        start_time = time.time()
        for i in range(N):
            centroids[c_array[i]] += data[i]
        centroids = centroids / np.copy(cluster_sizes_arr).reshape(-1, 1)

        # print "Total variation", total_variation
        # print "Cluster sizes", cluster_sizes
        # print("--- %s seconds ---" % (time.time() - start_time))
        # print c
        # print centroids

    return total_variation, c_array

def kmeans_paralell(k, data, nr_iter=100):
    N = len(data)

    # Choose k random data points as centroids
    centroids = data[np.random.choice(
        np.array(range(N)), size=k, replace=False)]
    print "Initial centroids\n", centroids

    N = len(data)
    c_array = mp.Array("i", N)
    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    print "Iteration\tVariation\tDelta Variation"
    total_variation = 0.0
    partitioned = np.array_split(range(N), k)

    for j in range(nr_iter):
        #print "=== Iteration %d ===" % (j+1)

        # Assign data points to nearest centroid
        variation = mp.Array("d", k)
        cluster_sizes = mp.Array("i", k)
        start_time = time.time()
        workers = []

        for z in range(k):
            workers.append(mp.Process(target=nearestCentroid_paritioned_parallel,
                                      args=(partitioned[z], data, c_array, cluster_sizes, variation, centroids)))
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        delta_variation = -total_variation
        total_variation = sum(variation)
        delta_variation += total_variation
        print "%3d\t\t%f\t%f" % (j, total_variation, delta_variation)
        print("--- %s seconds ---" % (time.time() - start_time))

        cluster_sizes_arr = np.frombuffer(
            cluster_sizes.get_obj(), dtype="int32")
        # Recompute centroids
        centroids = np.zeros((k, 2))
        start_time = time.time()
        for i in range(N):
            centroids[c_array[i]] += data[i]
        centroids = centroids / np.copy(cluster_sizes_arr).reshape(-1, 1)

        # print "Total variation", total_variation
        # print "Cluster sizes", cluster_sizes
        # print("--- %s seconds ---" % (time.time() - start_time))
        # print c
        # print centroids

    return total_variation, c_array


if __name__ == "__main__":
    n_samples = 10000

    X = generateData(n_samples)
    start_time = time.time()
    kmeans_paralell(3, X, nr_iter=20)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    kmeans_serial(3, X, nr_iter=20)
    print("--- %s seconds ---" % (time.time() - start_time))
