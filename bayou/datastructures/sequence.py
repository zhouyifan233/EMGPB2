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
        self.n_components = initial_state.n_components
        self.filtered = np.empty([self.len], dtype=GMM)
        self.filtered_collapsed = np.empty([self.len], dtype=Gaussian)
        self.smoothed = np.empty([self.len], dtype=GMM)
        self.smoothed_collapsed = np.empty([self.len], dtype=Gaussian)
        # self.Pr_Stplus1_St_y1T = np.zeros([self.len], dtype=GMM)

        N = self.n_components
        dims = self.initial_state.gaussian_dims
        T = self.len

        filtered_crossvar = np.empty([N, N], dtype=np.ndarray)
        for m in range(N):
            for n in range(N):
                filtered_crossvar[m, n] = np.zeros([T, dims[n], dims[n]])
        self.filtered_crossvar = filtered_crossvar

        smoothed_crossvar = np.empty([N, N], dtype=np.ndarray)
        for m in range(N):
            for n in range(N):
                smoothed_crossvar[m, n] = np.zeros([T, dims[n], dims[n]])
        self.smoothed_crossvar = smoothed_crossvar
        self.smoothed_crossvar_collapsed = np.zeros((T, dims[0], dims[0]))

        self.loglikelihood = np.zeros([T, N, N])
        self.measurement_likelihood = np.zeros([T])

        # Pr(s_{t-1} = j, s_{t} = k | y_{1:t})
        # self.filter_joint_pr = np.zeros([T, N, N])
        # Pr(s_{t-1} = j, s_{t} = k | y_{1:T})
        # self.smooth_joint_pr = np.zeros([T, N, N])
        self.smooth_j_k_t = np.empty([self.len, N, N], dtype=Gaussian)

    def get_filter_crossvar_time(self, t):
        N = self.initial_state.n_components
        VV = np.empty([N, N], dtype=np.ndarray)
        for j in range(N):
            for k in range(N):
                VV[j, k] = self.filtered_crossvar[j, k][t]
        return VV

    def get_smooth_crossvar_time(self, t):
        N = self.initial_state.n_components
        VV = np.empty([N, N], dtype=np.ndarray)
        for j in range(N):
            for k in range(N):
                VV[j, k] = self.smoothed_crossvar[j, k][t]
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

    def get_filtered_means(self, t):
        mu = []
        for i in range(self.n_components):
            mu.append(self.filtered[t].components[i].mean)
        return np.array(mu)

    def get_smoothed_means(self, t):
        mu = []
        for i in range(self.n_components):
            mu.append(self.smoothed[t].components[i].mean)
        return np.array(mu)

    def get_smooth_weights(self):
        weights = []
        for t in range(self.len):
            weights.append(self.smoothed[t].weights)
        return np.array(weights)

    def get_n_components(self):
        return self.n_components

    def get_smothed_Pr_Stplus1_St_y1T(self):
        Pr_Stplus1_St_y1T = []
        for t in range(self.len):
            Pr_Stplus1_St_y1T.append(self.smoothed[t].Pr_Stplus1_St_y1T)
        return np.asarray(Pr_Stplus1_St_y1T)
