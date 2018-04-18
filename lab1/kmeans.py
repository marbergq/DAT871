import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import multiprocessing as mp
import time
from ctypes import c_double

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

def nearestCentroid_batched(partitioned, data, centroids,variation,c, cluster_sizes):
    for i in partitioned:
        cluster, dist = nearestCentroid(data[i], centroids)
        c[i] = cluster
        cluster_sizes[cluster] += 1
        variation[cluster] += dist**2
    print "done with",partitioned[0]
    return

def kmeans_parallel(k, data, nr_iter = 100):
    N = len(data)

    # Choose k random data points as centroids
    centroids = data[np.random.choice(np.array(range(N)),size=k,replace=False)]
    print "Initial centroids\n", centroids

    N = len(data)
    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = mp.Array('i', np.zeros(N, dtype=int))

    print "Iteration\tVariation\tDelta Variation"
    total_variation = 0.0
    nwokkers=8
    partitioned = np.array_split(range(N), nwokkers)
    
    for j in range(nr_iter):
        #print "=== Iteration %d ===" % (j+1)

        # Assign data points to nearest centroid
        variation = mp.Array(c_double, k)
        cluster_sizes = mp.Array('i',k)
        workers = []

        start_time = time.time()

        for i in range(nwokkers):
            workers.append(mp.Process(target=nearestCentroid_batched, args=(partitioned[i], data, centroids,variation,c, cluster_sizes)))
        for w in workers:
            w.start()
        for w in workers:
            w.join()
            w.terminate()

        print("--- %s seconds ---" % (time.time() - start_time))
            
        delta_variation = -total_variation
        total_variation = sum(np.frombuffer(variation.get_obj())) 
        delta_variation += total_variation
        print "%3d\t\t%f\t%f" % (j, total_variation, delta_variation)

        # Recompute centroids
        centroids = np.zeros((k,2))
        for i in range(N):
            centroids[c[i]] += data[i]   
        sizes= np.frombuffer(cluster_sizes.get_obj(), dtype=np.int32)
        centroids = centroids / sizes.reshape(-1,1)
        
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
        nearestCentroid_batched(range(N),data,centroids,variation,c,cluster_sizes) 
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
    n_samples = 100000

    X = generateData(n_samples)
    start_time = time.time()
    kmeans_parallel(3, X, nr_iter=20)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    kmeans(3, X, nr_iter=20)
    print("--- %s seconds ---" % (time.time() - start_time))