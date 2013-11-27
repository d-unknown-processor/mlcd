__author__ = 'arenduchintala'

import random

NUM_TOPICS = 5  # K
alpha = 0.1
beta = 0.01
lamb = 0.5
nk = {}
VOCAB = {}
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


"""
   options = [0, 1, 2, 3]
    prob = [0.1, 0.0, 0.0, 0.9]
    counts = {o: 0 for o in options}
    for t in range(1000):
        counts[multinomial_sampling(options, prob)] += 1
    pprint(counts)
"""