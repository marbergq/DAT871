from mrjob.job import MRJob
import numpy as np
import itertools
import time


class LAB2Statistics(MRJob):

    def mapper(self, _, line):
        (_, _, value) = line.split()
        yield ("value", float(value))
        # yield (int((float(value)*100) % 10), 1)

    def combiner(self, key, values):
        if key == "value":
            listValue = list(values)
            yield ("max", max(listValue))
            yield ("min", min(listValue))
            yield ("values", listValue)

    def reducer(self, key, values):
        if key == "max":
            yield (key, max(values))
        elif key == "min":
            yield (key, min(values))
        elif key == "values":
            listValues = list(itertools.chain(*values))
            yield ("std", np.std(listValues))
            yield ("median", np.median(listValues))
            yield ("mean", np.mean(listValues))


if __name__ == '__main__':
    start_time = time.time()
    LAB2Statistics.run()
    # print "Runtime : ", time.time()-start_time
