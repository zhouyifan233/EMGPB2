# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian


class Sequence():
    """Base Sequence

    Attributes
    ----------
    len : no of measurements
    measurements : np.ndarray
        (len, dim, 1)
    """
    def __init__(self, measurements):
        self.len = measurements.shape[0]
        self.measurements = measurements


class GaussianSequence(Sequence):
    """ Sequence of Gaussian States

    Attributes
    ----------
    len : no of measurements
    measurements : np.ndarray
        (len, dim, 1)
    initial_state : Gaussian State
    filtered : np.ndarray of  filtered Gaussian States
    smoothed : np.ndarray of smoothed Gaussian States
    filter_crossvar : np.ndarray
        first dimension is the time index, 2nd and 3rd dimension make up cross variance matrix.
    loglikelihood : np.ndarray
    """

    def __init__(self, measurements, initial_state):
        super().__init__(measurements)
        if type(initial_state) != Gaussian:
            raise ValueError(
                "initial state should be a gaussian"
            )
        self.initial_state = initial_state
        self.filtered = np.empty([self.len], dtype=Gaussian)
        self.smoothed = np.empty([self.len], dtype=Gaussian)
        self.filter_crossvar = np.zeros([self.len, self.initial_state.dim, self.initial_state.dim])
        self.smooth_crossvar = np.zeros([self.len, self.initial_state.dim, self.initial_state.dim])
        self.loglikelihood = np.zeros([self.len])
