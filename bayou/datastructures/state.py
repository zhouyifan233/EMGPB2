# -*- coding: utf-8 -*-
import numpy as np
from bayou.utils import Utility


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

    def __init__(self, mean, covar):
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

        return Utility.Collapse(components, weights, transforms)

        '''
        x = 0.0
        V = 0.0

        for n in range(n_components):
            x += np.exp(weights[n]) * transforms[n] @ components[n].mean

        for n in range(n_components):
            diff = transforms[n] @ components[n].mean - x
            V += np.exp(weights[n]) * (transforms[n] @ components[n].covar @ transforms[n].T + diff @ diff.T)

        return Gaussian(mean=x, covar=V)
        '''
