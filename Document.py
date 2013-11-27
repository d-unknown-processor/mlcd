__author__ = 'arenduchintala'

from globals import *


class Document():
    def __init__(self, tokens, cd, z, x):
        self.tokens = tokens
        self.corpus = cd
        self.z = z
        self.x = x
        self.nd = {'*': len(tokens)}
        self.theta = {}
        self.initial_document_counts()

    def initial_document_counts(self):
        for k in range(NUM_TOPICS):
            self.nd[k] = self.z.count(k)

    def exclude_document_topic_counts(self, current_zdi):
        """
        here we update self.nd every time step 2.a.i is being done
        """
        self.nd[current_zdi] -= 1
        self.nd['*'] -= 1
        if self.nd[current_zdi] < 0 or self.nd['*'] < 0:
            print current_zdi, self.nd[current_zdi], self.nd['*']
            raise ValueError('nd_k or nd_star can not be less than 0')


    def check_document_topic_counts(self):
        sum_ks = 0.0
        for k in range(NUM_TOPICS):
            sum_ks = sum_ks + self.nd[k]
        if sum_ks != self.nd['*']:
            raise BaseException('the sum of ndks is not equal to sum of nd_star!')


    def include_document_topic_counts(self, new_zdi):
        """
        update counts of self.nd every time step 2.a.iv is being done
        """
        self.nd[new_zdi] += 1
        self.nd['*'] += 1
        if self.nd[new_zdi] > len(self.tokens) or self.nd['*'] > len(self.tokens):
            print new_zdi, self.nd[new_zdi], self.nd['*'], len(self.tokens)
            raise ValueError('nd_k or nd_star can not be greater than lenght of document')


    def get_new_zdi(self, token, corpus=None):
        """
        selects a new_new_zdi based on sampling weights for each k
        Note: the appropriate phi_kw should be passed in.
        phi_kw should be the global version if xdi = 0
        phi_kw should be the corpus version if xdi = 1
        """
        sampling_weights = [0.0 for _ in range(NUM_TOPICS)]
        for k in range(NUM_TOPICS):
            theta_dk = (self.nd[k] + alpha) / (self.nd['*'] + NUM_TOPICS * alpha)
            # compute phi based on value of argument corpus, if corpus is None, use general phi, else use corpus specific phi
            if corpus is None:
                nkw = nk.get(('global', k, token), 0.0)
                nk_star = nk[('global', k, '*')]
                phi_kw = (nkw + beta) / (nk_star + len(VOCAB) * beta)
            else:
                nckw = nk.get((corpus, k, token), 0.0)
                nck_star = nk[(corpus, k, '*')]
                phi_kw = (nckw + beta) / (nck_star + len(VOCAB) * beta)

            if corpus is None:
                #print 'geting new zdi', k, nkw, nk_star
                if nk_star < 0 or nkw < 0:
                    raise BaseException(' global counts gone below zero')
            else:
                #print 'geting new zdi', k, nckw, nck_star, corpus
                if nck_star < 0 or nckw < 0:
                    raise BaseException(' corpus counts gone below zero')

            sampling_weights[k] = theta_dk * phi_kw

        sum_weight = float(sum(sampling_weights))
        probability_per_k = [w / sum_weight for w in sampling_weights]
        #new_zdi = list(np.random.multinomial(1, probability_per_k, 1)[0]).index(1)
        new_zdi = multinomial_sampling(range(NUM_TOPICS), probability_per_k)
        return new_zdi


    def get_new_xdi(self, zdi, token, corpus):
        """
        returns a new_xdi based on phi_zdi_w
        weight for new_xdi == 0 is (1-lambda)*phi_zdi_w
        weight for new_xdi == 1 is (lambda)*phi_zdi_w
        phi_zdi_w should be appropriately selected
        """
        nkw = nk.get(('global', zdi, token), 0.0)
        nk_star = nk.get(('global', zdi, '*'), 0.0)
        phi_zdi_w = ( nkw + beta) / ( + len(VOCAB) * beta)

        nckw = nk.get((corpus, zdi, token), 0.0)
        nck_star = nk[(corpus, zdi, '*')]
        phi_c_zdi_w = (nckw + beta) / ( nck_star + len(VOCAB) * beta)

        sample_weights = [(1 - lamb) * phi_zdi_w, lamb * phi_c_zdi_w]
        sum_weight = float(sum(sample_weights))
        probability_per_binary = [w / sum_weight for w in sample_weights]
        #new_xdi = list(np.random.multinomial(1, probability_per_binary, 1)[0]).index(1)
        new_xdi = multinomial_sampling([0, 1], probability_per_binary)
        return new_xdi


    def compute_theta(self):
        for k in range(NUM_TOPICS):
            self.theta[k] = (self.nd[k] + alpha) / (self.nd['*'] + NUM_TOPICS * alpha)

        if abs(sum(self.theta.values()) - 1.0) > 1e-15:
            raise BaseException('sum of document topic thetas is not 1.0')

