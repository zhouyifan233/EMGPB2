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

    def __init__(self, mean, covar):
        self.dim = mean.shape[0]
        self.mean = mean
        self.covar = covar
        if self.dim != self.covar.shape[0]:
            raise ValueError(
                "mean and covar should have the same dimension"
            )
