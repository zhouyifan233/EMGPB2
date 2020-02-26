# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy.special import logsumexp
from emgpb2.states import Gaussian, GMM
from emgpb2.utils import Utility


class KalmanFilter:
    """ """

    @staticmethod
    def filter(state, measurement, model, initial=False, shape=None):

        # Transition Model
        A = model.A
        Q = model.Q
        # Measurement Model
        H = model.H
        R = model.R

        if shape is None:
            shape = np.eye(state.dim)

        x = shape @ state.mean
        V = shape @ state.covar @ shape.T

        y = measurement

        if initial:
            x_predict = x
            V_predict = V
        else:
            x_predict = A @ x
            V_predict = A @ V @ A.T + Q

        y_predict = H @ x_predict
        y_predict_covar = H @ V_predict @ H.T + R

        error = y - y_predict
        filter_gain = V_predict @ H.T @ linalg.inv(y_predict_covar)

        n_dim_state = x.shape[0]

        new_x = x_predict + filter_gain @ error
        new_V = (np.eye(n_dim_state) - filter_gain @ H) @ V_predict
        # new_V = (new_V + new_V.T) / 2.0  # Force Symmetric

        # V_t_tminus1
        if initial:
            V_new_old = new_V
        else:
            V_new_old = (np.eye(n_dim_state) - filter_gain @ H) @ A @ V
        # V_new_old = (V_new_old + V_new_old.T) / 2.0  # Force Symmetric

        # log likelihood
        L = Utility.get_log_gaussian_prob(error, np.zeros_like(error), y_predict_covar)
        # L = Utility.get_log_gaussian_prob((new_x - x_predict), np.zeros_like(new_x), V_predict)

        return Gaussian(mean=new_x, covar=new_V), V_new_old, L

    @staticmethod
    def filter_sequence(sequence, model):

        for t in range(0, sequence.len):
            if t == 0:
                state, VV, L = KalmanFilter.filter(sequence.initial_state,
                                             sequence.measurements[t],
                                             model,
                                             True)
            else:
                state, VV, L = KalmanFilter.filter(sequence.filtered[t - 1],
                                             sequence.measurements[t],
                                             model,
                                             False)
            sequence.filtered[t] = state
            sequence.filter_crossvar[t] = VV
            sequence.loglikelihood[t] = L

        return sequence


class GPB2Filter:
    """ """

    @staticmethod
    def filter(gmm_state, measurement, models, Z, M_tminus1, initial=False):
        N = gmm_state.n_components
        filtered_i_j_t = np.empty([N, N], dtype=Gaussian)
        VV_i_j_t_tminus1 = np.empty([N, N], dtype=np.ndarray)
        L_i_j_t = np.ones([N, N])

        for i in range(N):
            for j in range(N):
                (filtered_i_j_t[i, j], VV_i_j_t_tminus1[i, j], L_i_j_t[i, j]) = KalmanFilter.filter(gmm_state.components[i],
                                                                        measurement,
                                                                        models[j],
                                                                        initial,
                                                                        gmm_state.transforms[i, j])

        # tmp_numerator = np.exp(L_i_j_t) * Z * M_tminus1     # i * j
        tmp_numerator = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                tmp_numerator[i, j] = np.exp(L_i_j_t[i, j]) * Z[i, j] * M_tminus1[i]

        if np.any(tmp_numerator == 0):
            print(tmp_numerator)
        measurement_likelihood = logsumexp(np.log(tmp_numerator))       # 1
        M_tminus1_t = tmp_numerator / np.sum(tmp_numerator)     # i, j
        M_t = np.sum(M_tminus1_t, axis=0)       # j
        W = np.zeros((N, N))      # i, j
        for i in range(N):
            for j in range(N):
                W[i, j] = M_tminus1_t[i, j] / M_t[j]

        states_j = []
        for j in range(N):
            state_j = Utility.Collapse(components=list(filtered_i_j_t[:, j]),
                                       weights=list(W[:, j]),
                                       transforms=[np.eye(gmm_state.gaussian_dims[j])] * N)
            states_j.append(state_j)

        new_gmm_state = GMM(states_j)
        # print(M_t)
        new_gmm_state.weights = M_t

        return new_gmm_state, VV_i_j_t_tminus1, L_i_j_t, M_t, measurement_likelihood

    @staticmethod
    def filter_sequence(gmmsequence, models, Z):
        n_components = gmmsequence.initial_state.n_components
        for t in range(0, gmmsequence.len):
            if t == 0:
                M_t = np.ones(n_components) / n_components
                gmm_state, VV, LL, M_t, yL = GPB2Filter.filter(gmmsequence.initial_state,
                                                         gmmsequence.measurements[t],
                                                         models,
                                                         Z,
                                                         M_t,
                                                         True)
            else:
                gmm_state, VV, LL, M_t, yL = GPB2Filter.filter(gmmsequence.filtered[t - 1],
                                                         gmmsequence.measurements[t],
                                                         models,
                                                         Z,
                                                         M_t,
                                                         False)
            gmmsequence.filtered[t] = gmm_state
            gmmsequence.filtered_collapsed[t] = gmm_state.collapse()
            # This is just rearrange the cross variance of filtering process.
            for j in range(n_components):
                for k in range(n_components):
                    gmmsequence.filtered_crossvar[j, k][t] = VV[j, k]
            gmmsequence.loglikelihood[t] = LL
            gmmsequence.measurement_likelihood[t] = yL

        return gmmsequence
