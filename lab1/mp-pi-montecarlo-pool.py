import multiprocessing
import argparse # See https://docs.python.org/2/howto/argparse.html for a tutorial
import random
from math import pi

def sample_pi(n):
    """ Perform n steps of Monte Carlo simulation for estimating Pi/4.
        Returns the number of sucesses."""
    random.seed()
    s = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            s += 1
    return s

def worker(batch_size, queue):
    while True:
        s = sample_pi(batch_size)
        queue.put(s)

def compute_pi(args):
    batch_size = 1000
    results = multiprocessing.Queue()
    workers = []

    for _ in range(args.workers):
        workers.append(multiprocessing.Process(target=worker, args=(batch_size, results)))
    for w in workers:
        w.start()

    n_total = 0
    s_total = 0
    pi_est = 0
    accuracy = 1 / float(args.accuracy)
    
    while abs(pi-pi_est) > accuracy:
        s_total += results.get()
        n_total += batch_size
        pi_est = (4.0*s_total)/n_total
    
    for w in workers:
        w.terminate()
        w.join()
    
    print " Steps\tSuccess\tPi est.\tError"
    print "%6d\t%7d\t%1.10f\t%1.10f" % (n_total, s_total, pi_est, pi-pi_est)

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
