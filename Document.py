__author__ = 'arenduchintala'

import globals
from sys import stderr


class Document():
    def __init__(self, tokens, cd, z, x):
        self.tokens = tokens
        self.corpus = cd
        self.z = z
        self.x = x
        self.nd = {'*': len(tokens)}
        self.theta = {}
        self.theta_burn_in = {}
        self.initial_document_counts()

    def initial_document_counts(self):
        for k in range(globals.NUM_TOPICS):
            self.nd[k] = self.z.count(k)

    def exclude_document_topic_counts(self, current_zdi):
        """
        here we update self.nd every time step 2.a.i is being done
        """
        self.nd[current_zdi] -= 1
        self.nd['*'] -= 1
        if not globals.DIAGNOSTICS:
            return True
        if self.nd[current_zdi] < 0 or self.nd['*'] < 0:
            print current_zdi, self.nd[current_zdi], self.nd['*']
            raise ValueError('nd_k or nd_star can not be less than 0')


    def check_document_topic_counts(self):
        if not globals.DIAGNOSTICS:
            return True
        sum_ks = 0.0
        for k in range(globals.NUM_TOPICS):
            sum_ks = sum_ks + self.nd[k]
        if sum_ks != self.nd['*']:
            raise BaseException('the sum of ndks is not equal to sum of nd_star!')


    def include_document_topic_counts(self, new_zdi):
        """
        update counts of self.nd every time step 2.a.iv is being done
        """
        self.nd[new_zdi] += 1
        self.nd['*'] += 1
        if not globals.DIAGNOSTICS:
            return True
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
        sampling_weights = [0.0 for _ in range(globals.NUM_TOPICS)]
        for k in range(globals.NUM_TOPICS):
            theta_dk = (self.nd[k] + globals.ALPHA) / (self.nd['*'] + globals.NUM_TOPICS * globals.ALPHA)
            # compute phi based on value of argument corpus, if corpus is None, use general phi, else use corpus specific phi
            if corpus is None:
                nkw = globals.nk.get(('global', k, token), 0.0)
                nk_star = globals.nk[('global', k, '*')]
                phi_kw = (nkw + globals.BETA) / (nk_star + len(globals.ALL_VOCAB) * globals.BETA)
            else:
                nckw = globals.nk.get((corpus, k, token), 0.0)
                nck_star = globals.nk[(corpus, k, '*')]
                phi_kw = (nckw + globals.BETA) / (nck_star + len(globals.ALL_VOCAB) * globals.BETA)

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
        new_zdi = globals.multinomial_sampling(range(globals.NUM_TOPICS), probability_per_k)
        return new_zdi


    def get_new_xdi(self, zdi, token, corpus):
        """
        returns a new_xdi based on phi_zdi_w
        weight for new_xdi == 0 is (1-lambda)*phi_zdi_w
        weight for new_xdi == 1 is (lambda)*phi_zdi_w
        phi_zdi_w should be appropriately selected
        """
        nkw = globals.nk.get(('global', zdi, token), 0.0)
        nk_star = globals.nk.get(('global', zdi, '*'), 0.0)
        phi_zdi_w = ( nkw + globals.BETA) / ( nk_star + len(globals.ALL_VOCAB) * globals.BETA)

        nckw = globals.nk.get((corpus, zdi, token), 0.0)
        nck_star = globals.nk[(corpus, zdi, '*')]
        phi_c_zdi_w = (nckw + globals.BETA) / ( nck_star + len(globals.ALL_VOCAB) * globals.BETA)

        sample_weights = [(1 - globals.LAMBDA) * phi_zdi_w, globals.LAMBDA * phi_c_zdi_w]
        sum_weight = float(sum(sample_weights))
        probability_per_binary = [w / sum_weight for w in sample_weights]
        #new_xdi = list(np.random.multinomial(1, probability_per_binary, 1)[0]).index(1)
        new_xdi = globals.multinomial_sampling([0, 1], probability_per_binary)
        return new_xdi


    def compute_theta(self, burn_in_passed=False):
        for k in range(globals.NUM_TOPICS):
            self.theta[k] = (self.nd[k] + globals.ALPHA) / (self.nd['*'] + globals.NUM_TOPICS * globals.ALPHA)

        if burn_in_passed:
            for key in self.theta:
                self.theta_burn_in[key] = self.theta_burn_in.get(key, 0.0) + self.theta[key]

        if not globals.DIAGNOSTICS:
            return True
        if abs(sum(self.theta.values()) - 1.0) > 1e-5:
            print 'difference:', abs(sum(self.theta.values()) - 1.0)
            raise BaseException('sum of document topic thetas is not 1.0')


    def get_new_zdi_as_test_document(self, token, current_phi, corpus=None):
        sampling_weights = [0.0 for _ in range(globals.NUM_TOPICS)]
        for k in range(globals.NUM_TOPICS):
            theta_dk = (self.nd[k] + globals.ALPHA) / (self.nd['*'] + globals.NUM_TOPICS * globals.ALPHA)
            if corpus is None:
                phi = current_phi[('global', k, token)]
            else:
                phi = current_phi[(corpus, k, token)]
            sampling_weights[k] = theta_dk * phi

        sum_weight = float(sum(sampling_weights))
        if sum_weight == 0:
            raise BaseException(str('Sum of weights for token: ' + token + ' is 0.0, this must be an unseen word'))
        elif sum_weight < 0:
            raise BaseException('Sum of weights is negative, this is a problem!')
        else:
            pass
        probability_per_k = [w / sum_weight for w in sampling_weights]
        #new_zdi = list(np.random.multinomial(1, probability_per_k, 1)[0]).index(1)
        new_zdi = globals.multinomial_sampling(range(globals.NUM_TOPICS), probability_per_k)
        return new_zdi


    def get_new_xdi_as_test_document(self, zdi, token, corpus, current_phi):
        phi_zdi_w = current_phi[('global', zdi, token)]
        phi_c_zdi_w = current_phi[(corpus, zdi, token)]

        sample_weights = [(1 - globals.LAMBDA) * phi_zdi_w, globals.LAMBDA * phi_c_zdi_w]
        sum_weight = float(sum(sample_weights))
        if sum_weight == 0:
            raise BaseException(str('Sum of weights for token: ' + token + ' is 0.0, this must be an unseen word'))
        elif sum_weight < 0:
            raise BaseException('Sum of weights is negative, this is a problem!')
        else:
            pass
        probability_per_binary = [w / sum_weight for w in sample_weights]
        #new_xdi = list(np.random.multinomial(1, probability_per_binary, 1)[0]).index(1)
        new_xdi = globals.multinomial_sampling([0, 1], probability_per_binary)
        return new_xdi
