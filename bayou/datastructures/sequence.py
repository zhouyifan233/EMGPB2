# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GMM


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


class GMMSequence(Sequence):
    """ Sequence of Gaussian Mixture Models

    Attributes
    ----------
    len : no of measurements
    measurements : np.ndarray
        (len, dim, 1)
    initial_state : GMM State
    filtered : np.ndarray of  filtered GMM States
    smoothed : np.ndarray of smoothed GMM States
    filter_crossvar : np.ndarray of np.ndarrays
        (no_components, no_components) array of (len, , ) arrays
    """

    def __init__(self, measurements, initial_state):
        super().__init__(measurements)
        if type(initial_state) != GMM:
            raise ValueError(
                "initial state should be a gmm"
            )
        self.initial_state = initial_state
        self.filtered = np.empty([self.len], dtype=GMM)
        self.smoothed = np.empty([self.len], dtype=GMM)

        N = self.initial_state.n_components
        dims = self.initial_state.gaussian_dims
        T = self.len

        filter_crossvar = np.empty([N, N], dtype=np.ndarray)
        for m in range(N):
            for n in range(N):
                filter_crossvar[m, n] = np.zeros([T, dims[n], dims[n]])
        self.filter_crossvar = filter_crossvar

        smooth_crossvar = np.empty([N, N], dtype=np.ndarray)
        for m in range(N):
            for n in range(N):
                smooth_crossvar[m,n] = np.zeros([T, dims[n], dims[n]])
        self.smooth_crossvar = smooth_crossvar

        self.loglikelihood = np.zeros([T, N, N])
        self.measurement_likelihood = np.zeros([T])

        # Pr(s_{t-1} = j, s_{t} = k | y_{1:t})
        self.filter_joint_pr = np.zeros([T, N, N])
        # Pr(s_{t-1} = j, s_{t} = k | y_{1:T})
        self.smooth_joint_pr = np.zeros([T, N, N])

        self.smooth_j_k_t = np.empty([self.len, N, N], dtype=Gaussian)

    def get_filter_crossvar(self, t):
        N = self.initial_state.n_components
        VV = np.empty([N, N], dtype=np.ndarray)
        for j in range(N):
            for k in range(N):
                VV[j, k] = self.filter_crossvar[j, k][t]
        return VV

    def get_smooth_crossvar(self, t):
        N = self.initial_state.n_components
        VV = np.empty([N, N], dtype=np.ndarray)
        for j in range(N):
            for k in range(N):
                VV[j, k] = self.smooth_crossvar[j, k][t]
        return VV

    def get_filter_estimates(self):
        state_estimates = []
        for t in range(self.len):
            state_estimates.append(self.filtered[t].collapse().mean)
        return state_estimates

    def get_smooth_estimates(self):
        state_estimates = []
        for t in range(self.len):
            state_estimates.append(self.smoothed[t].collapse().mean)
        return state_estimates

    def get_smooth_weights(self, m):
        weights = []
        for t in range(self.len):
            weights.append(self.smoothed[t].weights[m])
        return np.array(weights)
