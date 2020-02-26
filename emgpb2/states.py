# -*- coding: utf-8 -*-
import numpy as np


class State():
    """Base State"""


class Gaussian(State):
    """ Gaussian State type

    Attributes
    ----------
    dim : int
    mean : np.ndarray
        (dim, 1)
    covar : np.ndarray
        (dim, dim)
    """

    def __init__(self, mean: np.ndarray, covar: np.ndarray):
        self.dim = mean.shape[0]
        self.mean = mean
        self.covar = covar
        if self.dim != self.covar.shape[0]:
            raise ValueError(
                "mean and covar should have the same dimension"
            )


class GMM(State):
    """ Gaussian Mixture Model State type

    Attributes
    ----------
    n_components : int
    gaussian_dims : list of ints
    components : list of Gaussian objects
    weights : np.ndarray
        (n_components, 1) log weights
    transforms : np.ndarray
        (n_components, n_components)
    """

    def __init__(self, gaussian_list):
        self.n_components = len(gaussian_list)
        self.gaussian_dims = [g.dim for g in gaussian_list]
        self.components = gaussian_list
        self.weights = np.log(np.ones([self.n_components, 1]) / self.n_components)
        self.Pr_Stplus1_St_y1T = np.zeros((self.n_components, self.n_components))

        transforms = []
        for i in self.gaussian_dims:
            row = []
            for j in self.gaussian_dims:
                transform = np.zeros([j, i])
                np.fill_diagonal(transform, 1)
                row.append(transform)
            transforms.append(row)
        self.transforms = np.array(transforms)

    def collapse(self, components=None, weights=None, transforms=None):
        """
        Parameters
        ----------

        weights : np.array
            probability weights
        """
        if components is None:
            components = self.components
        if weights is None:
            weights = self.weights
        if transforms is None:
            transforms = self.transforms[:, 0]
        n_components = len(components)
        dim = components[0].mean.shape[0]

        x = np.zeros((dim, 1))
        V = np.zeros((dim, dim))

        for n in range(n_components):
            x += weights[n] * transforms[n] @ components[n].mean

        for n in range(n_components):
            diff = transforms[n] @ components[n].mean - x
            V += weights[n] * (transforms[n] @ components[n].covar @ transforms[n].T + diff @ diff.T)

        return Gaussian(mean=x, covar=V)


class Sequence:
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

    def get_smoothed_Pr_Stplus1_St_y1T(self):
        Pr_Stplus1_St_y1T = []
        for t in range(self.len):
            Pr_Stplus1_St_y1T.append(self.smoothed[t].Pr_Stplus1_St_y1T)
        return np.asarray(Pr_Stplus1_St_y1T)

