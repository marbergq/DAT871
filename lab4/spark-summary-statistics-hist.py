import random
from pyspark import SparkContext
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
    return (v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2], min(v1[3], v2[3]), max(v1[3], v2[3]))

def mapAgain(v):
    print("key:",v[0], "value", v[1])
    return v[0], v[1]


result = sc.textFile("test.data") \
    .map(mapper) \
    .reduceByKey(combiner) \
    .map(mapAgain) \
    .reduceByKey(combiner) \
    .collect()

print("contents ", result)
