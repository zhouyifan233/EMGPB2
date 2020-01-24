# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg

from bayou.datastructures import Gaussian
from bayou.smoothers.base import Smoother


class LinearGaussian(Smoother):
    """ """


class RTS(LinearGaussian):
    """ """

    @staticmethod
    def smooth(smoothed_state_tplus1, filtered_state_t, filtered_state_tplus1, V_tplus1_t, model, shape=None):
        """ Forward-backward smoother for linear Gaussian state space model.
        """
        # Transition Model
        A = model.A
        Q = model.Q
        # Measurement Model
        H = model.H
        R = model.R

        if shape is None:
            shape = np.eye(filtered_state_t.dim)

        x = shape @ filtered_state_t.mean
        V = shape @ filtered_state_t.covar @ shape.T

        x_predict = A @ x
        V_predict = A @ V @ A.T + Q

        x_tplus1 = smoothed_state_tplus1.mean
        V_tplus1 = smoothed_state_tplus1.covar

        smoother_gain = V @ A.T @ linalg.inv(V_predict)

        x_smoothed = x + smoother_gain @ (x_tplus1 - x_predict)
        V_smoothed = V + smoother_gain @ (V_tplus1 - V_predict) @ smoother_gain.T

        #V_f_tplus1 = filtered_state_tplus1.covar
        #V_smoothed_tplus1_t = V_f_tplus1 @ smoother_gain.T + smoother_gain @ (V_tplus1_t - A @ V_f_tplus1) @ smoother_gain.T

        V_f_tplus1 = filtered_state_tplus1.covar
        V_smoothed_tplus1_t = (
            V_tplus1_t + (V_tplus1 - V_f_tplus1) @ linalg.inv(V_f_tplus1) @ V_tplus1_t
        )

        return Gaussian(mean=x_smoothed, covar=V_smoothed), V_smoothed_tplus1_t

    @staticmethod
    def smooth_sequence(sequence, model):
        penultimate_index = sequence.len - 2
        sequence.smoothed[-1] = sequence.filtered[-1]

        # Iterating backwards from the penultimate state, to the first state.
        for t in range(penultimate_index, -1, -1):
            state, smooth_VV = RTS.smooth(sequence.smoothed[t + 1],
                                          sequence.filtered[t],
                                          sequence.filtered[t + 1],
                                          sequence.filter_crossvar[t + 1],
                                          model)
            sequence.smoothed[t] = state
            sequence.smooth_crossvar[t+1] = smooth_VV

        return sequence
