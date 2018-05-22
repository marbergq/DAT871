""" Example solution to Assignment 2, Problem 1

    Run with python mrjob-summary-statistcs.py -r local summarydata.data

    mrjob is available in Conda.
"""

from mrjob.job import MRJob, MRStep
import mrjob
from mrjob.protocol import JSONValueProtocol
import math

class SummaryStatistics(MRJob):

    INPUT_PROTOCOL = mrjob.protocol.TextProtocol 
    # So  <key>\t<remainder of line> is passed as key, value to mapper

    # This is called once before calling the mappers
    def mapper_init(self):
        # This should be read from the command line and 
        self.nr_bins = 10
        self.v_min = 0.0
        self.v_max = 1.0
        self.binwidth = (self.v_max - self.v_min) / self.nr_bins

    def mapper(self, key, value):
        # Value is the part of the line following the first \t
        s, v = value.split('\t')
        secondary = int(s)
        value = float(v)

        # Determine the bin value belongs too
        if value < self.v_min:
            bin = 0
        elif value > self.v_max:
            bin = self.nr_bins - 1
        else:
            bin = int((value - self.v_min) / self.binwidth)
        yield bin, value 


    # Combiner runs on single node and aggregates data with the same key
    # Here: all the values extracted by mapper
    def combiner(self, bin, values):
        s = sos = 0.0 

        # Note: values is an iterator and not a list, so we cannot write sum(values), min(values),...,
        # as iterators produce values only once
        for n, v in enumerate(values):
            if n == 0:
                v_min = v
                v_max = v
            s += v
            sos += v*v
            v_min = min(v_min, v)
            v_max = max(v_max, v)

        yield bin, (n+1, s, sos, v_min, v_max)


    def reducer(self, bin, stats):
        # There will be exactly one reducer call per bin.
        perbin_n = 0
        perbin_sum = 0.0
        perbin_sumofsquares = 0.0

        for i, s in enumerate(stats):
            if i == 0:
                perbin_min = s[3]
                perbin_max = s[4]

            perbin_n += s[0]
            perbin_sum += s[1]
            perbin_sumofsquares += s[2]
            perbin_min = min(perbin_min, s[3])
            perbin_max = max(perbin_max, s[4])

        yield "stats", (bin, perbin_n, perbin_sum, perbin_sumofsquares, perbin_min, perbin_max)


    def finalreducer(self, key, stats):
        # There will be exactly one reducer getting all ("stats", ...) tuples
        total_n = 0
        total_sum = 0.0
        total_sumofsquares = 0.0

        counts = {}
        
        for i, s in enumerate(stats):
            if i == 0:
                total_min = s[4]
                total_max = s[5]
            bin = s[0]
            counts[bin] = s[1]
            total_n += s[1]
            total_sum += s[2]
            total_sumofsquares += s[3]
            total_min = min(total_min, s[4])
            total_max = max(total_max, s[5])

        bins = counts.keys()
        bins.sort()

        mean = total_sum/total_n
        variance = total_sumofsquares / total_n - mean**2
        stddev = math.sqrt(variance)
        yield "n", total_n
        yield "mean", mean
        yield "stddev", stddev
        yield "min", total_min
        yield "max", total_max
        yield "hist", [counts[b] for b in bins]
        

    def steps(self):
         return [MRStep(mapper_init=self.mapper_init,
                        mapper=self.mapper,
                        combiner=self.combiner,
                        reducer=self.reducer),
            MRStep(reducer=self.finalreducer)]   


if __name__ == '__main__':
    SummaryStatistics.run()
