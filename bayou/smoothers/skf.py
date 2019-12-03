# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import logsumexp

from bayou.smoothers.base import Smoother
from bayou.datastructures import Gaussian, GMM
from bayou.smoothers.lineargaussian import RTS
from bayou.utils import Utility


class SKF(Smoother):
    """ """


class GPB2(SKF):
    """ """

    @staticmethod
    def smooth(smooth_gmm_state_tplus1,
               filtered_gmm_state_t,
               filtered_gmm_state_tplus1,
               VV_j_k_tplus1,
               models,
               Z):
        N = smooth_gmm_state_tplus1.n_components
        smooth_j_k_t = np.empty([N, N], dtype=Gaussian)
        smooth_VV_j_k_tplus1 = np.empty([N, N], dtype=np.ndarray)

        for j in range(N):
            for k in range(N):
                (smooth_j_k_t[j, k],
                 smooth_VV_j_k_tplus1[j, k]) = RTS.smooth(smooth_gmm_state_tplus1.components[k],
                                                          filtered_gmm_state_t.components[j],
                                                          filtered_gmm_state_tplus1.components[k],
                                                          VV_j_k_tplus1[j, k],
                                                          models[k],
                                                          smooth_gmm_state_tplus1.transforms[j, k])

        U = filtered_gmm_state_t.weights + np.log(Z)
        U = U - logsumexp(U, axis=0, keepdims=True)
        log_pr_t_tplus1_T = U + smooth_gmm_state_tplus1.weights.T

        log_weights_t = logsumexp(log_pr_t_tplus1_T, axis=1, keepdims=True)
        weights_t = log_weights_t

        W_kj = log_pr_t_tplus1_T - log_weights_t
        W = W_kj

        states_t = []
        for j in range(N):
            state_j = smooth_gmm_state_tplus1.collapse(components=smooth_j_k_t[j, :],
                                                       weights=W[j, :],
                                                       transforms=smooth_gmm_state_tplus1.transforms[:, j])
            states_t.append(state_j)

        new_gmm_state = GMM(states_t)
        new_gmm_state.weights = weights_t

        return new_gmm_state, smooth_j_k_t, smooth_VV_j_k_tplus1, log_pr_t_tplus1_T

    @staticmethod
    def smooth_sequence(gmmsequence, models, Z):
        penultimate_index = gmmsequence.len - 2
        gmmsequence.smoothed[-1] = gmmsequence.filtered[-1]

        for t in range(penultimate_index, -1, -1):
            gmm_state, state_j_k_t, VV, jPr = GPB2.smooth(gmmsequence.smoothed[t + 1],
                                                          gmmsequence.filtered[t],
                                                          gmmsequence.filtered[t + 1],
                                                          gmmsequence.get_filter_crossvar(t + 1),
                                                          models,
                                                          Z)

            gmmsequence.smoothed[t] = gmm_state
            N = gmmsequence.initial_state.n_components
            for j in range(N):
                for k in range(N):
                    gmmsequence.smooth_crossvar[j, k][t + 1] = VV[j, k]
                    gmmsequence.smooth_j_k_t[t, j, k] = state_j_k_t[j, k]
            gmmsequence.smooth_joint_pr[t + 1] = jPr

        return gmmsequence
