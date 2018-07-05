# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg


class Utility():
    """Base utility class"""

    @staticmethod
    def get_log_gaussian_prob(x, mean, covar):
        dim = x.shape[0]
        denom = ((2 * np.pi) ** (dim / 2.)) * np.sqrt(np.abs(linalg.det(covar)))
        mahal = np.sum(((x - mean).T @ linalg.inv(covar)) * (x - mean).T)  # Possibly better??
        return -0.5 * mahal - np.log(denom)

    @staticmethod
    def check_lik_convergence(new_lik, old_lik, threshold=0.0001):
        diff = np.abs(new_lik - old_lik)
        avg = (np.abs(new_lik) + np.abs(old_lik) + np.spacing(1)) / 2.
        if (diff / avg) < threshold:
            return True
        return False
