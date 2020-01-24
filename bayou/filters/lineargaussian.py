# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg

from bayou.datastructures import Gaussian
from bayou.filters.base import Filter
from bayou.utils.util import Utility


class LinearGaussian(Filter):
    """ """


class Kalman(LinearGaussian):
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

        return Gaussian(mean=new_x, covar=new_V), V_new_old, L

    @staticmethod
    def filter_sequence(sequence, model):

        for t in range(0, sequence.len):
            if t == 0:
                state, VV, L = Kalman.filter(sequence.initial_state,
                                             sequence.measurements[t],
                                             model,
                                             True)
            else:
                state, VV, L = Kalman.filter(sequence.filtered[t - 1],
                                             sequence.measurements[t],
                                             model,
                                             False)
            sequence.filtered[t] = state
            sequence.filter_crossvar[t] = VV
            sequence.loglikelihood[t] = L

        return sequence
