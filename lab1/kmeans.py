import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import multiprocessing as mp
import time

def generateData(n, plot=None):
    X, y = make_blobs(n_samples=n, cluster_std=1.7, shuffle=False, random_state = 2122)
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

def nearestCentroid_batched(inqueue, queue):
    while True:
        request = inqueue.get()
        i = request[0]
        data = request[1]
        centroids = request[2]
        cluster, dist = nearestCentroid(data, centroids)
        queue.put((i,cluster,dist))

def kmeans_parallel(k, data, nr_iter = 100):
    N = len(data)

    # Choose k random data points as centroids
    centroids = data[np.random.choice(np.array(range(N)),size=k,replace=False)]
    print "Initial centroids\n", centroids

    N = len(data)
    results = mp.Queue(N+1)
    jobs = mp.Queue()
    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)

    print "Iteration\tVariation\tDelta Variation"
    total_variation = 0.0
    
    workers = []

    for i in range(8):
        workers.append(mp.Process(target=nearestCentroid_batched, args=(jobs,results)))
    for w in workers:
        w.start()

    for j in range(nr_iter):
        #print "=== Iteration %d ===" % (j+1)

        # Assign data points to nearest centroid
        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int) 

        for i in range(N):
            jobs.put((i, data[i], centroids))

        for i in range(N):
            result = results.get()
            i = result[0]
            cluster = result[1]
            dist = result[2]
            c[i] = cluster
            cluster_sizes[cluster] += 1
            variation[cluster] += dist**2

        delta_variation = -total_variation
        total_variation = sum(variation) 
        delta_variation += total_variation
        print "%3d\t\t%f\t%f" % (j, total_variation, delta_variation)

        # Recompute centroids
        centroids = np.zeros((k,2))
        for i in range(N):
            centroids[c[i]] += data[i]        
        centroids = centroids / cluster_sizes.reshape(-1,1)
        
        #print "Total variation", total_variation
        #print "Cluster sizes", cluster_sizes
        #print c
        #print centroids
    
    return total_variation, c

def kmeans(k, data, nr_iter = 100):
    N = len(data)

    # Choose k random data points as centroids
    centroids = data[np.random.choice(np.array(range(N)),size=k,replace=False)]
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
            cluster, dist = nearestCentroid(data[i],centroids)
            c[i] = cluster
            cluster_sizes[cluster] += 1
            variation[cluster] += dist**2
        delta_variation = -total_variation
        total_variation = sum(variation) 
        delta_variation += total_variation
        print "%3d\t\t%f\t%f" % (j, total_variation, delta_variation)

        # Recompute centroids
        centroids = np.zeros((k,2))
        for i in range(N):
            centroids[c[i]] += data[i]        
        centroids = centroids / cluster_sizes.reshape(-1,1)
        
        #print "Total variation", total_variation
        #print "Cluster sizes", cluster_sizes
        #print c
        #print centroids
    
    return total_variation, c

if __name__ == "__main__":
    n_samples = 10000

    X = generateData(n_samples)
    start_time = time.time()
    kmeans_parallel(3, X, nr_iter=20)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    kmeans(3, X, nr_iter=20)
    print("--- %s seconds ---" % (time.time() - start_time))