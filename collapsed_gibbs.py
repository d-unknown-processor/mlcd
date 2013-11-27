__author__ = 'arenduchintala'

import random
from sys import stderr
from globals import *
from Document import Document
from math import log
from collections import defaultdict
from pprint import pprint

training_documents = []
testing_documents = []


def initialize_nk():
    for d in training_documents:
        current_corpus = d.corpus
        for idx, token in enumerate(d.tokens):
            zdi = d.z[idx]
            nk[('global', zdi, token)] = nk.get(('global', zdi, token), 0.0) + 1.0
            nk[('global', zdi, '*')] = nk.get(('global', zdi, '*'), 0.0) + 1.0
            nk[(current_corpus, zdi, token)] = nk.get((current_corpus, zdi, token), 0.0) + 1.0
            nk[(current_corpus, zdi, '*')] = nk.get((current_corpus, zdi, '*'), 0.0) + 1.0
            VOCAB[token] = VOCAB.get(token, 0.0) + 1.0


def check():
    print 'checking...'
    sum_global_stars = 0.0
    for k in range(NUM_TOPICS):
        acl_k = nk[('ACL', k, '*')]
        nips_k = nk[('NIPS', k, '*')]
        global_k = nk[('global', k, '*')]
        sum_global_stars += global_k
        print k, 'global', global_k, ' NIPS', nips_k, ' ACL', acl_k, 'passed=', global_k == nips_k + acl_k
        assert global_k == nips_k + acl_k
    print 'number of tokens in corpus', sum_global_stars, '(this should not change across iterations)'


def include_topic_token_counts(zdi, token, corpus):
    if ('global', zdi, token) in nk:
        nk[('global', zdi, token)] += 1.0
    else:
        #print 'new key found!!', ('global', zdi, token)
        nk[('global', zdi, token)] = 1.0

    if ('global', zdi, '*') in nk:
        nk[('global', zdi, '*')] += 1.0
    else:
        #print 'new key found!!', ('global', zdi, '*')
        nk[('global', zdi, '*')] = 1.0

    if (corpus, zdi, token) in nk:
        nk[(corpus, zdi, token)] += 1.0
    else:
        #print 'new key found!!', (corpus, zdi, token)
        nk[(corpus, zdi, token)] = 1.0

    if (corpus, zdi, '*') in nk:
        nk[(corpus, zdi, '*')] += 1.0
    else:
        #print 'new key found!!', (corpus, zdi, '*')
        nk[(corpus, zdi, '*')] = 1.0


def exclude_topic_token_counts(zdi, token, corpus):
    """
    update nk so that zdi is currently ignored
    """
    if ('global', zdi, token) in nk:
        #print 'reducing ', ('global', zdi, token), ' form ', nk[('global', zdi, token)]
        nk[('global', zdi, token)] -= 1.0
        #print 'reducing ', ('global', zdi, '*'), ' form ', nk[('global', zdi, '*')]
        nk[('global', zdi, '*')] -= 1.0

    if (corpus, zdi, token) in nk:
        #print 'reducing ', (corpus, zdi, token), ' form ', nk[(corpus, zdi, token)]
        nk[(corpus, zdi, token)] -= 1.0
        #print 'reducing ', (corpus, zdi, '*'), ' form ', nk[(corpus, zdi, '*')]
        nk[(corpus, zdi, '*')] -= 1.0


def compute_phi():
    current_phi = {}  # includes both global phi and corpus specific phi
    for (corp, k, token) in nk:
        if token != '*': # what if i don't have this check?
            current_phi[(corp, k, token)] = (nk[(corp, k, token)] + beta) / (nk[(corp, k, '*')] + len(VOCAB) * beta)
    return current_phi


def log_likelihood(document_set, current_phi):
    sum_log_likelihood = 0.0
    for d in document_set:
        for idx, w in enumerate(d.tokens):
            inner = 0.0
            for z in range(NUM_TOPICS):
                phi_zw = current_phi.get(('global', z, w), 0.0)
                phi_cd_zw = current_phi.get((d.corpus, z, w), 0.0)
                inner += d.theta[z] * ((1 - lamb) * phi_zw + lamb * phi_cd_zw)
            sum_log_likelihood += log(inner)
    return sum_log_likelihood


def iterate():
    for i, d in enumerate(training_documents):
        current_corpus = d.corpus
        for idx, token in enumerate(d.tokens):
            zdi = d.z[idx]
            xdi = d.x[idx]
            # step 2.a.i
            d.exclude_document_topic_counts(zdi)
            exclude_topic_token_counts(zdi, token, current_corpus)

            # step 2.a.ii
            if xdi == 0:
                new_zdi = d.get_new_zdi(token)
            else:
                new_zdi = d.get_new_zdi(token, current_corpus)

            # step 2.a.iii
            new_xdi = d.get_new_xdi(new_zdi, token, current_corpus)

            # step 2.a.iv
            d.z[idx] = new_zdi
            d.x[idx] = new_xdi

            d.include_document_topic_counts(new_zdi)
            include_topic_token_counts(new_zdi, token, current_corpus)

        d.compute_theta()
        d.check_document_topic_counts()
    current_phi = compute_phi()
    #check()
    return current_phi


if __name__ == '__main__':

    training_data = open('hw3-files/input-train.txt', 'r').readlines()
    testing_data = open('hw3-files/input-test.txt', 'r').readlines()

    for doc_id, document in enumerate(training_data):
        tokens = document.strip().split()
        cd = int(tokens.pop(0))
        corpora = 'NIPS' if cd == 0 else 'ACL'
        Z = [random.randint(0, NUM_TOPICS - 1) for i in xrange(len(tokens))]
        X = [random.randint(0, 1) for i in xrange(len(tokens))]
        d = Document(tokens, corpora, Z, X)
        training_documents.append(d)

    for doc_id, document in enumerate(testing_data):
        tokens = document.strip().split()
        cd = int(tokens.pop(0))
        corpora = 'NIPS' if cd == 0 else 'ACL'
        Z = [random.randint(0, NUM_TOPICS - 1) for i in xrange(len(tokens))]
        X = [random.randint(0, 1) for i in xrange(len(tokens))]
        d = Document(tokens, corpora, Z, X)
        testing_documents.append(d)

    initialize_nk()
    check()
    for t in range(100):
        stderr.write('ITERATION ' + str(t) + '\n')
        current_phi = iterate()
        #pprint(current_phi)
        print 'log-likelihood', log_likelihood(training_documents, current_phi)








