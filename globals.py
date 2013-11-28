__author__ = 'arenduchintala'

import random

#defaults
NUM_TOPICS = 5
ALPHA = 0.1
BETA = 0.01
LAMBDA = 0.5
TRAIN_FILE = 'hw3-files/input-train.txt'
TEST_FILE = 'hw3-files/input-test.txt'
OUT_FILE = 'output.txt'
BURN_IN = 100
NUM_ITERATIONS = 110

#internal
nk = {}
ALL_VOCAB = {}
UNSEEN_VOCAB = {}
DIAGNOSTICS = False


def multinomial_sampling(options, probabilities):
    cp = [sum(probabilities[:i + 1]) for i in xrange(len(probabilities))]
    r = random.random()
    option_idx = filter(lambda x: cp[x] > r > cp[x - 1], range(len(cp)))
    if len(option_idx) == 0:
        return options[0]
    else:
        return options[option_idx[0]]


def parse_params(args):
    global TRAIN_FILE
    TRAIN_FILE = args[1]
    global TEST_FILE
    TEST_FILE = args[2]
    global OUT_FILE
    OUT_FILE = args[3]
    global NUM_TOPICS
    NUM_TOPICS = int(args[4])
    print NUM_TOPICS, ' is the num topics'
    global LAMBDA
    LAMBDA = float(args[5])
    global ALPHA
    ALPHA = float(args[6])
    global BETA
    BETA = float(args[7])
    global NUM_ITERATIONS
    NUM_ITERATIONS = int(args[8])
    global BURN_IN
    BURN_IN = int(args[9])


"""
   options = [0, 1, 2, 3]
    prob = [0.1, 0.0, 0.0, 0.9]
    counts = {o: 0 for o in options}
    for t in range(1000):
        counts[multinomial_sampling(options, prob)] += 1
    pprint(counts)
"""