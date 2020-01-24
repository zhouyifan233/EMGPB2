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
               VV_j_k_tplus1_t,
               models,
               Z):
        N = smooth_gmm_state_tplus1.n_components
        smoothed_j_k_t = np.empty([N, N], dtype=Gaussian)
        smoothed_VV_j_k_tplus1_t = np.empty([N, N], dtype=np.ndarray)

        for j in range(N):
            for k in range(N):
                (smoothed_j_k_t[j, k],
                 smoothed_VV_j_k_tplus1_t[j, k]) = RTS.smooth(smooth_gmm_state_tplus1.components[k],
                                                          filtered_gmm_state_t.components[j],
                                                          filtered_gmm_state_tplus1.components[k],
                                                          VV_j_k_tplus1_t[j, k],                    # Swapped
                                                          models[k],
                                                          smooth_gmm_state_tplus1.transforms[j, k])
        # get M_{t|t}
        M_t = filtered_gmm_state_t.weights
        # calculate M_{t|t}(j)Z(j, k)
        tmp_numerator = np.zeros((N, N))
        for j in range(N):
            for k in range(N):
                tmp_numerator[j, k] = M_t[j] * Z[j, k]
        # calculate U^{j|k}_{t}
        U = np.zeros((N, N))        # j * k
        for j in range(N):
            for k in range(N):
                U[j, k] = tmp_numerator[j, k] / np.sum(tmp_numerator[:, k])
        # get M_{t+1|T}
        M_tplus1 = smooth_gmm_state_tplus1.weights
        # calculate M_{t,t+1|T}
        M_t_tplus1 = np.zeros((N, N))       # j * k
        for j in range(N):
            for k in range(N):
                M_t_tplus1[j, k] = U[j, k] * M_tplus1[k]
        # calculate M_{t|T}
        M_t_smoothed = np.sum(M_t_tplus1, axis=1)       # j
        # calculate W^{k|j}_{t}
        W_t = np.zeros((N, N))      # j, k
        for j in range(N):
            for k in range(N):
                W_t[j, k] = M_t_tplus1[j, k] / M_t_smoothed[j]
        # calculate (x^{j}_{t|T} , V^{j}_{t|T})
        states_t = []
        for j in range(N):
            state_j = Utility.Collapse(components=smoothed_j_k_t[j, :],
                                       weights=W_t[j, :],
                                       transforms=smooth_gmm_state_tplus1.transforms[:, j])
            states_t.append(state_j)
        new_gmm_state = GMM(states_t)
        new_gmm_state.weights = M_t_smoothed
        new_gmm_state.Pr_Stplus1_St_y1T = W_t #     for estimating Z
        # calculate x^{j(k)}_{t+1|T} this is an approximated approach
        x_j_k_tplus1_t = np.empty((N, N), dtype=np.ndarray)
        for j in range(N):
            for k in range(N):
                x_j_k_tplus1_t[j, k] = smooth_gmm_state_tplus1.components[k].mean
        # calculate x^{(j)k}_{t|T}
        x_j_k_t = np.empty((N, N), dtype=np.ndarray)
        for j in range(N):
            for k in range(N):
                x_j_k_t[j, k] = smoothed_j_k_t[j, k].mean
        # calculate V^{k}_{t+1,t|T}
        smoothed_VV_k_tplus1_t = []
        for k in range(N):
            VV_k = Utility.CollapseCross(list(x_j_k_tplus1_t[:, k]),
                                         list(x_j_k_t[:, k]),
                                         list(smoothed_VV_j_k_tplus1_t[:, k]),
                                         list(U[:, k]))
            smoothed_VV_k_tplus1_t.append(VV_k)
        # calculate x^{()k}_{t|T}
        x_k_t = np.empty(N, dtype=np.ndarray)
        for k in range(N):
            tmp = 0.0
            for j in range(N):
                tmp += smoothed_j_k_t[j, k].mean * U[j, k]
            x_k_t[k] = tmp

        return new_gmm_state, smoothed_VV_j_k_tplus1_t, (smoothed_VV_k_tplus1_t, x_k_t, M_tplus1)

    @staticmethod
    def smooth_sequence(gmmsequence, models, Z):
        penultimate_index = gmmsequence.len - 2
        gmmsequence.smoothed[-1] = gmmsequence.filtered[-1]
        gmmsequence.smoothed_collapsed[-1] = gmmsequence.filtered_collapsed[-1]

        for t in range(penultimate_index, -1, -1):
            gmm_state, VV_j_k_tplus1_t, params_ = GPB2.smooth(gmmsequence.smoothed[t + 1],
                                        gmmsequence.filtered[t],
                                        gmmsequence.filtered[t + 1],
                                        gmmsequence.get_filter_crossvar_time(t + 1),
                                        models,
                                        Z)

            gmmsequence.smoothed[t] = gmm_state
            # calculate (x_{t|T} , V_{t|T})
            smoothed_t = gmm_state.collapse()
            gmmsequence.smoothed_collapsed[t] = smoothed_t

            # calculate V_{t+1,t|T}
            x_k_tplus1 = gmmsequence.get_smoothed_means(t+1)
            smoothed_VV_k_tplus1_t, x_k_t, M_tplus1 = params_
            gmmsequence.smoothed_crossvar_collapsed[t + 1] = Utility.CollapseCross(x_k_tplus1, x_k_t, smoothed_VV_k_tplus1_t, M_tplus1)

            N = gmmsequence.initial_state.n_components
            for j in range(N):
                for k in range(N):
                    gmmsequence.smoothed_crossvar[j, k][t + 1] = VV_j_k_tplus1_t[j, k]

        return gmmsequence
