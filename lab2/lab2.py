from mrjob.job import MRJob
import numpy as np


class LAB2Statistics(MRJob):
    def mapper(self, _, line):
        (_, _, value) = line.split()
        yield ("value", float(value))
        yield (int((float(value)*100) % 10), 1)

    def combiner(self, key, values):
        if key == "value":
            listValue = list(values)
            yield ("max", max(listValue))
            yield ("min", min(listValue))
            yield ("avg", sum(listValue)/len(listValue))
        else:
            yield (key, sum(values))
    
    def reducer(self, key, values):
        if key == "max" : 
            yield (key, max(values))
        elif key == "min":
            yield (key, min(values))
        elif key == "avg":
            listavgs = list(values)
            yield (key, sum(listavgs)/len(listavgs))
        else :
            yield (key, sum(values))
if __name__ == '__main__':
    LAB2Statistics.run()
