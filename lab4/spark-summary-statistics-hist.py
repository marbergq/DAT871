import random
from pyspark import SparkContext
import math
sc = SparkContext()
nr_bins = 10
v_min = 0.0
v_max = 1.0
binwidth = (v_max - v_min) / nr_bins


def mapper(value):
    splitted = value.split('\t')
    value = float(splitted[2])
    if value < v_min:
        bin = 0
    elif value > v_max:
        bin = nr_bins - 1
    else:
        bin = int((value - v_min) / binwidth)

    # print((bin, value))
    return bin, (1, value, value**2, value, value)


def combiner(v1, v2):
    # print(v1, v2)
    return (v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2], min(v1[3], v2[3]), max(v1[4], v2[4]))


def finalReducer(v1, v2):
    return (v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2], min(v1[3], v2[3]), max(v1[4], v2[4]), v1[5]+v2[5])


def finalMap(v):
    bin, stats = v[0:]
    return stats+([(bin, stats[0])],)


n, mean, stddev, total_min, total_max, hist = sc.textFile("test.data") \
    .map(mapper) \
    .reduceByKey(combiner) \
    .map(finalMap) \
    .reduce(finalReducer)

mean = (mean/n)
stddev = math.sqrt(stddev / n - mean**2)

print("n", n)
print("mean", mean/n)
print("stddev", stddev)
print("min", total_min)
print("max", total_max)
print("hist", sorted(hist))
