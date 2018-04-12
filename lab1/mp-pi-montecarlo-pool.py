import multiprocessing
import argparse # See https://docs.python.org/2/howto/argparse.html for a tutorial
import random
from math import pi

def sample_pi(id, queue):
    """ Perform n steps of Monte Carlo simulation for estimating Pi/4.
        Returns the number of sucesses."""
    random.seed()
    while True:
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            queue.put(1)
        else:
            queue.put(0)

def compute_pi(args):
    results = multiprocessing.Queue()
    workers = []

    for id in range(args.workers):
        workers.append(multiprocessing.Process(target=sample_pi, args=(id,results)))
    for w in workers:
        w.start()

    n_total = 0
    s_total = 0
    pi_est = 0

    while abs(pi-pi_est) > 1 / float(args.accuracy):
        s_total += results.get()
        n_total += 1
        pi_est = (4.0*s_total)/n_total
    
    for w in workers:
        w.terminate()
        w.join()
    
    print " Steps\tSuccess\tPi est.\tError"
    print "%6d\t%7d\t%1.5f\t%1.5f" % (n_total, s_total, pi_est, pi-pi_est)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Pi using Monte Carlo simulation.')
    parser.add_argument('--workers', '-w',
                        default='1',
                        type = int,
                        help='Number of parallel processes')
    parser.add_argument('--accuracy', '-a',
                        default='1000',
                        type = int,
                        help='The accuracy')
    args = parser.parse_args()
    compute_pi(args)
