__author__ = 'arenduchintala'

import random
from sys import stderr, argv
import globals

from Document import Document
from math import log
from collections import defaultdict
from pprint import pprint
import pdb

training_documents = []
testing_documents = []

burn_in_phi = {}


def initialize():
    for d in training_documents:
        current_corpus = d.corpus
        for idx, token in enumerate(d.tokens):
            zdi = d.z[idx]
            xdi = d.x[idx]
            if xdi == 0:
                globals.nk[('global', zdi, token)] = globals.nk.get(('global', zdi, token), 0.0) + 1.0
                globals.nk[('global', zdi, '*')] = globals.nk.get(('global', zdi, '*'), 0.0) + 1.0
            else:
                globals.nk[(current_corpus, zdi, token)] = globals.nk.get((current_corpus, zdi, token), 0.0) + 1.0
                globals.nk[(current_corpus, zdi, '*')] = globals.nk.get((current_corpus, zdi, '*'), 0.0) + 1.0
            globals.ALL_VOCAB[token] = globals.ALL_VOCAB.get(token, 0.0) + 1.0

    for d in testing_documents:
        for token in d.tokens:
            if token not in globals.ALL_VOCAB:
                globals.UNSEEN_VOCAB[token] = globals.UNSEEN_VOCAB.get(token, 0.0) + 1.0
            globals.ALL_VOCAB[token] = globals.ALL_VOCAB.get(token, 0.0) + 1.0


def check():
    if not globals.DIAGNOSTICS:
        return True
    print 'checking...'
    sum_global_stars = 0.0
    for k in range(globals.NUM_TOPICS):
        acl_k = globals.nk[('ACL', k, '*')]
        nips_k = globals.nk[('NIPS', k, '*')]
        global_k = globals.nk[('global', k, '*')]
        sum_global_stars += global_k
        print k, 'global', global_k, ' NIPS', nips_k, ' ACL', acl_k, 'passed=', global_k == nips_k + acl_k
        assert global_k == nips_k + acl_k
    print 'number of tokens in corpus', sum_global_stars, '(this should not change across iterations)'


def include_topic_token_counts(zdi, token, corpus):
    if ('global', zdi, token) in globals.nk:
        globals.nk[('global', zdi, token)] += 1.0
    else:
        #print 'new key found!!', ('global', zdi, token)
        globals.nk[('global', zdi, token)] = 1.0

    if ('global', zdi, '*') in globals.nk:
        globals.nk[('global', zdi, '*')] += 1.0
    else:
        #print 'new key found!!', ('global', zdi, '*')
        globals.nk[('global', zdi, '*')] = 1.0

    if (corpus, zdi, token) in globals.nk:
        globals.nk[(corpus, zdi, token)] += 1.0
    else:
        #print 'new key found!!', (corpus, zdi, token)
        globals.nk[(corpus, zdi, token)] = 1.0

    if (corpus, zdi, '*') in globals.nk:
        globals.nk[(corpus, zdi, '*')] += 1.0
    else:
        #print 'new key found!!', (corpus, zdi, '*')
        globals.nk[(corpus, zdi, '*')] = 1.0


def exclude_topic_token_counts(zdi, token, corpus):
    """
    update globals.nk so that zdi is currently ignored
    """
    if ('global', zdi, token) in globals.nk:
        #print 'reducing ', ('global', zdi, token), ' form ', globals.nk[('global', zdi, token)]
        globals.nk[('global', zdi, token)] -= 1.0
        if globals.nk[('global', zdi, token)] < 0:
            globals.nk[('global', zdi, token)] = 0
            #print 'reducing ', ('global', zdi, '*'), ' form ', globals.nk[('global', zdi, '*')]
        globals.nk[('global', zdi, '*')] -= 1.0
        if globals.nk[('global', zdi, '*')] < 0:
            globals.nk[('global', zdi, '*')] = 0

    if (corpus, zdi, token) in globals.nk:
        #print 'reducing ', (corpus, zdi, token), ' form ', globals.nk[(corpus, zdi, token)]
        globals.nk[(corpus, zdi, token)] -= 1.0
        if globals.nk[(corpus, zdi, token)] < 0:
            globals.nk[(corpus, zdi, token)] = 0
            #print 'reducing ', (corpus, zdi, '*'), ' form ', globals.nk[(corpus, zdi, '*')]
        globals.nk[(corpus, zdi, '*')] -= 1.0
        if globals.nk[(corpus, zdi, '*')] < 0:
            globals.nk[(corpus, zdi, '*')] = 0


def compute_phi(burn_in_passed=False):
    current_phi = {}  # includes both global phi and corpus specific phi
    """
    for (corp, k, token) in globals.nk:
        if token != '*': # what if i don't have this check?
            current_phi[(corp, k, token)] = (globals.nk[(corp, k, token)] + beta) / (globals.nk[(corp, k, '*')] + len(VOCAB) * beta)
    """
    """
    for unseen_token in globals.UNSEEN_VOCAB:
        for k in range(globals.NUM_TOPICS):
            for corp in ['global', 'NIPS', 'ACL']:
                nckw = globals.nk.get((corp, k, unseen_token), 0.0)

                current_phi[(corp, k, unseen_token)] = (nckw + globals.BETA) / (globals.nk[(corp, k, '*')] + len(globals.ALL_VOCAB) * globals.BETA)

                if burn_in_passed:
                    burn_in_phi[(corp, k, unseen_token)] = burn_in_phi.get((corp, k, unseen_token), 0.0) + current_phi[
                        (corp, k, unseen_token)]
    """
    for token in globals.ALL_VOCAB:
        for k in range(globals.NUM_TOPICS):
            for corp in ['global', 'NIPS', 'ACL']:
                nckw = globals.nk.get((corp, k, token), 0.0)
                if nckw != 0 and token in globals.UNSEEN_VOCAB:
                    raise BaseException(str('corp:' + corp + ' topic:' + str(k) + ' token:' + token + ' is unseen but has counts'))
                current_phi[(corp, k, token)] = (nckw + globals.BETA) / (globals.nk[(corp, k, '*')] + len(globals.ALL_VOCAB) * globals.BETA)

                if burn_in_passed:
                    burn_in_phi[(corp, k, token)] = burn_in_phi.get((corp, k, token), 0.0) + current_phi[(corp, k, token)]

    return current_phi


def log_likelihood(document_set, current_phi):
    sum_log_likelihood = 0.0
    for d in document_set:
        for idx, w in enumerate(d.tokens):
            inner = 0.0
            for z in range(globals.NUM_TOPICS):
                phi_zw = current_phi[('global', z, w)]
                phi_cd_zw = current_phi[(d.corpus, z, w)]
                inner += d.theta[z] * ((1 - globals.LAMBDA) * phi_zw + globals.LAMBDA * phi_cd_zw)
            if inner > 0.0:
                sum_log_likelihood += log(inner)
    return sum_log_likelihood


def iterate_test_set(t, current_phi):
    for i, d in enumerate(testing_documents):
        current_corpus = d.corpus
        for idx, token in enumerate(d.tokens):
            zdi = d.z[idx]
            xdi = d.x[idx]
            d.exclude_document_topic_counts(zdi)
            if xdi == 0:
                new_zdi = d.get_new_zdi_as_test_document(token, current_phi)
            else:
                new_zdi = d.get_new_zdi_as_test_document(token, current_phi, current_corpus)
            new_xdi = d.get_new_xdi_as_test_document(new_zdi, token, current_corpus, current_phi)

            d.z[idx] = new_zdi
            d.x[idx] = new_xdi

            d.include_document_topic_counts(new_zdi)
        d.compute_theta(t > globals.BURN_IN - 1)
        d.check_document_topic_counts()


def iterate_train_set(t):
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

        d.compute_theta(t > globals.BURN_IN - 1)
        d.check_document_topic_counts()
    current_phi = compute_phi(t > globals.BURN_IN - 1)
    return current_phi


def write_sample_mean_phi(burn_in_phi):
    writable_phi = defaultdict(list)
    for w in globals.ALL_VOCAB:
        for corp in ['global', 'NIPS', 'ACL']:
            phi_ks = []
            for k in range(globals.NUM_TOPICS):
                phi_ks.append("%.13e" % (burn_in_phi[(corp, k, w)] / float(globals.NUM_ITERATIONS - globals.BURN_IN)))
            line = w + ' ' + ' '.join(phi_ks)
            writable_phi[corp].append(line)
    do_writing(writable_phi['global'], globals.OUT_FILE + '-phi')
    do_writing(writable_phi['NIPS'], globals.OUT_FILE + '-phi0')
    do_writing(writable_phi['ACL'], globals.OUT_FILE + '-phi1')


def write_sample_mean_thetas():
    burns = globals.NUM_ITERATIONS - globals.BURN_IN
    all_d_sample_means = []
    for d in testing_documents:
        theta_d_sample_mean = ["%.13e" % (d.theta_burn_in[theta_dk] / float(burns)) for theta_dk in d.theta_burn_in]
        theta_d_sample_mean_str = ' '.join(theta_d_sample_mean)
        all_d_sample_means.append(theta_d_sample_mean_str)
    do_writing(all_d_sample_means, globals.OUT_FILE + '-theta')


def do_writing(a_list, file_name):
    stderr.write(str('writing ' + file_name + '...\n'))
    writer = open(file_name, 'w')
    writer.write('\n'.join(a_list))
    writer.flush()
    writer.close()


if __name__ == '__main__':

    globals.parse_params(argv[:])
    print "Params:"
    print globals.TRAIN_FILE, globals.TEST_FILE, globals.OUT_FILE, globals.NUM_TOPICS, globals.LAMBDA, globals.ALPHA, globals.BETA, globals.NUM_ITERATIONS, globals.BURN_IN
    training_data = open(globals.TRAIN_FILE, 'r').readlines()
    testing_data = open(globals.TEST_FILE, 'r').readlines()

    for doc_id, document in enumerate(training_data):
        tokens = document.strip().split()
        cd = int(tokens.pop(0))
        corpora = 'NIPS' if cd == 0 else 'ACL'
        Z = [random.randint(0, globals.NUM_TOPICS - 1) for i in xrange(len(tokens))]
        X = [random.randint(0, 1) for i in xrange(len(tokens))]
        d = Document(tokens, corpora, Z, X)
        training_documents.append(d)

    for doc_id, document in enumerate(testing_data):
        tokens = document.strip().split()
        cd = int(tokens.pop(0))
        corpora = 'NIPS' if cd == 0 else 'ACL'
        Z = [random.randint(0, globals.NUM_TOPICS - 1) for i in xrange(len(tokens))]
        X = [random.randint(0, 1) for i in xrange(len(tokens))]
        d = Document(tokens, corpora, Z, X)
        testing_documents.append(d)

    initialize()
    check()
    train_ll_list = []
    test_ll_list = []
    print len(globals.ALL_VOCAB)
    for t in range(globals.NUM_ITERATIONS):
        stderr.write('ITERATION ' + str(t) + '\n')
        current_phi = iterate_train_set(t)
        iterate_test_set(t, current_phi)
        check()
        tr_ll = "%.13e" % log_likelihood(training_documents, current_phi)
        te_ll = "%.13e" % log_likelihood(testing_documents, current_phi)
        train_ll_list.append(tr_ll)
        test_ll_list.append(te_ll)
        print 'train set log-likelihood', tr_ll
        print 'test  set log-likelihood', te_ll

    do_writing(train_ll_list, globals.OUT_FILE + '-trainll')

    do_writing(test_ll_list, globals.OUT_FILE + '-testll')

    write_sample_mean_thetas()
    write_sample_mean_phi(burn_in_phi)






