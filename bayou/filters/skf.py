# -*- coding: utf-8 -*-
import copy

import numpy as np
from scipy.special import logsumexp

from bayou.filters.base import Filter
from bayou.datastructures import Gaussian, GMM
from bayou.filters.lineargaussian import Kalman
#from bayou.resamplers import StratifiedResampler
from bayou.utils import Utility

import time


class SKF(Filter):
    """ """


class GPB2(SKF):
    """ """

    @staticmethod
    def filter(gmm_state, measurement, models, Z, initial=False):
        N = gmm_state.n_components
        filtered_j_k_tplus1 = np.empty([N, N], dtype=Gaussian)
        VV_j_k_tplus1 = np.empty([N, N], dtype=np.ndarray)
        LL = np.ones([N, N])

        for j in range(N):
            for k in range(N):
                (filtered_j_k_tplus1[j, k],
                 VV_j_k_tplus1[j, k],
                 LL[j, k]) = Kalman.filter(gmm_state.components[j],
                                           measurement,
                                           models[k],
                                           initial,
                                           gmm_state.transforms[j, k])

        I = LL + np.log(Z) + gmm_state.weights
        normalisation_constant = logsumexp(I)
        measurement_likelihood = normalisation_constant

        log_pr_t_tplus1_tplus1 = I - normalisation_constant

        log_weights_tplus1 = logsumexp(log_pr_t_tplus1_tplus1, axis=0, keepdims=True).T
        weights_tplus1 = log_weights_tplus1

        W = log_pr_t_tplus1_tplus1 - log_weights_tplus1.T

        states_tplus1 = []
        for k in range(N):
            state_k = gmm_state.collapse(components=filtered_j_k_tplus1[:, k],
                                         weights=W[:, k],
                                         transforms=[np.eye(gmm_state.gaussian_dims[k])] * N)
            states_tplus1.append(state_k)

        new_gmm_state = GMM(states_tplus1)
        new_gmm_state.weights = weights_tplus1

        return new_gmm_state, VV_j_k_tplus1, LL, log_pr_t_tplus1_tplus1, measurement_likelihood

    @staticmethod
    def filter_sequence(gmmsequence, models, Z):
        for t in range(0, gmmsequence.len):
            if t == 0:
                gmm_state, VV, LL, jPr, yL = GPB2.filter(gmmsequence.initial_state,
                                                         gmmsequence.measurements[t],
                                                         models,
                                                         Z,
                                                         True)
            else:
                gmm_state, VV, LL, jPr, yL = GPB2.filter(gmmsequence.filtered[t - 1],
                                                         gmmsequence.measurements[t],
                                                         models,
                                                         Z,
                                                         False)
            gmmsequence.filtered[t] = gmm_state
            N = gmmsequence.initial_state.n_components
            for j in range(N):
                for k in range(N):
                    gmmsequence.filter_crossvar[j, k][t] = VV[j, k]
            gmmsequence.loglikelihood[t] = LL
            gmmsequence.filter_joint_pr[t] = jPr
            gmmsequence.measurement_likelihood[t] = yL

        return gmmsequence
