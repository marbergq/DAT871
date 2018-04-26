import argparse
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data for HW 2, Problem 1.')
    parser.add_argument('--number', '-n',
                        default='1000',
                        type = int,
                        help='Number of records')
    args = parser.parse_args()

    n = args.number
    m = max(10, n / 100) 
    
    for i in xrange(n):
        primary = random.randint(0,n)
        secondary = random.randint(0,m)
        value = random.random()
        print "%d\t%d\t%f" %  (primary, secondary, value)

