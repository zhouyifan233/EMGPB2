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
    def filter(gmm_state, measurement, models, Z, M_tminus1, initial=False):
        N = gmm_state.n_components
        filtered_i_j_t = np.empty([N, N], dtype=Gaussian)
        VV_i_j_t_tminus1 = np.empty([N, N], dtype=np.ndarray)
        L_i_j_t = np.ones([N, N])

        for i in range(N):
            for j in range(N):
                (filtered_i_j_t[i, j], VV_i_j_t_tminus1[i, j], L_i_j_t[i, j]) = Kalman.filter(gmm_state.components[i],
                                                                        measurement,
                                                                        models[j],
                                                                        initial,
                                                                        gmm_state.transforms[i, j])

        # I = L_i_j_t + np.log(Z) + gmm_state.weights
        # normalisation_constant = logsumexp(I)
        # measurement_likelihood = normalisation_constant
        # log_pr_t_tplus1_tplus1 = I - normalisation_constant
        # log_weights_tplus1 = logsumexp(log_pr_t_tplus1_tplus1, axis=0, keepdims=True).T
        # weights_tplus1 = log_weights_tplus1
        # W = log_pr_t_tplus1_tplus1 - log_weights_tplus1.T
        # for i in range(N):
        #     for j in range(N):
        #        W[i, j] = M_tminus1_t[i, j] / M_t[j]

        tmp_numerator = np.exp(L_i_j_t) * Z * M_tminus1     # i * j
        measurement_likelihood = logsumexp(np.log(tmp_numerator))       # 1
        M_tminus1_t = tmp_numerator / np.sum(tmp_numerator)     # i, j
        M_t = np.sum(M_tminus1_t, axis=0)       # j
        W = np.zeros((N, N))      # i, j
        for i in range(N):
            for j in range(N):
                W[i, j] = M_tminus1_t[i, j] / M_t[j]

        states_j = []
        for j in range(N):
            # state_j = gmm_state.collapse(components=filtered_i_j_t[:, j],
            #                              weights=W[:, j],
            #                              transforms=[np.eye(gmm_state.gaussian_dims[j])] * N)
            state_j = Utility.Collapse(components=list(filtered_i_j_t[:, j]),
                                       weights=list(W[:, j]),
                                       transforms=[np.eye(gmm_state.gaussian_dims[j])] * N)
            states_j.append(state_j)

        new_gmm_state = GMM(states_j)
        new_gmm_state.weights = M_t

        return new_gmm_state, VV_i_j_t_tminus1, L_i_j_t, M_t, measurement_likelihood

    @staticmethod
    def filter_sequence(gmmsequence, models, Z):
        n_components = gmmsequence.initial_state.n_components
        for t in range(0, gmmsequence.len):
            if t == 0:
                M_t = np.ones(n_components) / n_components
                gmm_state, VV, LL, M_t, yL = GPB2.filter(gmmsequence.initial_state,
                                                         gmmsequence.measurements[t],
                                                         models,
                                                         Z,
                                                         M_t,
                                                         True)
            else:
                gmm_state, VV, LL, M_t, yL = GPB2.filter(gmmsequence.filtered[t - 1],
                                                         gmmsequence.measurements[t],
                                                         models,
                                                         Z,
                                                         M_t,
                                                         False)
            gmmsequence.filtered[t] = gmm_state
            # n_components = gmmsequence.initial_state.n_components
            # This is just rearrange the cross variance of filtering process.
            for j in range(n_components):
                for k in range(n_components):
                    gmmsequence.filter_crossvar[j, k][t] = VV[j, k]
            gmmsequence.loglikelihood[t] = LL
            # gmmsequence.filter_joint_pr[t] = jPr
            gmmsequence.measurement_likelihood[t] = yL

        return gmmsequence
