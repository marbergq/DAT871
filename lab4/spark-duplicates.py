import random
from pyspark import SparkContext
sc = SparkContext(master = 'local[4]')

distFile = sc.textFile("test.data")

counts = distFile.map(lambda l: l.split(',')) \
         .map(lambda t:(int(t[0]),1)) \
         .reduceByKey(lambda a, b: a + b) \
         .filter(lambda t:t[1] > 1)


cc = counts.collect()
print cc
