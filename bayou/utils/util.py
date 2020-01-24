# -*- coding: utf-8 -*-
import numpy as np
import copy
from scipy import linalg
from scipy.stats import multivariate_normal
from bayou.datastructures.state import Gaussian


class Utility():
    """Base utility class"""

    @staticmethod
    def get_log_gaussian_prob(x, mean, covar):
        dim = x.shape[0]
        denom = ((2 * np.pi) ** (dim / 2.)) * np.sqrt(np.abs(linalg.det(covar)))
        mahal = np.sum(((x - mean).T @ linalg.inv(covar)) * (x - mean).T)  # Possibly better??
        return -0.5 * mahal - np.log(denom)
        #log_prob = np.log(multivariate_normal.pdf(x, mean, covar))
        #return log_prob

    @staticmethod
    def check_lik_convergence(new_lik, old_lik, threshold=0.0001):
        diff = np.abs(new_lik - old_lik)
        #diff = old_lik - new_lik
        avg = (np.abs(new_lik) + np.abs(old_lik) + np.spacing(1)) / 2.
        if (diff / avg) < threshold:
            return True
        return False

    @staticmethod
    def stabilise_weights(weights, min_value=1e-6):
        """

        __OLD__ shouldn't need to use if the weights are kept in log

        weights = [n, 1] array like probabilities. (not log space)
        """
        indices_to_increase = [1 if v < min_value else 0 for i, v in enumerate(weights)]
        n_to_increase = sum(indices_to_increase)
        total_to_add = n_to_increase * min_value
        n_to_decrease = len(indices_to_increase) - n_to_increase
        to_subtract = total_to_add / n_to_decrease
        indices_to_decrease = [value ^ 1 for value in indices_to_increase]
        changes = min_value * np.array(indices_to_increase) - to_subtract * np.array(indices_to_decrease)
        return (weights.T + changes).T

    @staticmethod
    def normalise_logprob(lls_ori):
        lls = copy.copy(lls_ori)
        lls = np.asarray(lls)
        maxele = np.max(lls)
        minele = np.min(lls)
        for idx, thisll in enumerate(lls):
            lls[idx] = (thisll-minele)/(maxele-minele)
        new_probs = []
        for idex, thisll in enumerate(lls):
            new_probs.append(thisll / np.sum(lls))
        return new_probs

    @staticmethod
    def CollapseCross(x_list_t, x_list_tminus1, V_list_t_tminus1, W):
        n_components = len(W)
        dim = x_list_t[0].shape[0]
        mu_x_t = np.zeros((dim, 1))
        mu_x_tminus1 = np.zeros((dim, 1))

        for i in range(n_components):
            mu_x_t += W[i] * x_list_t[i]
            mu_x_tminus1 += W[i] * x_list_tminus1[i]

        V_collapsed_t_tminus1 = np.zeros_like(V_list_t_tminus1[0])
        for i in range(n_components):
            V_collapsed_t_tminus1 += W[i] * V_list_t_tminus1[i] + W[i] * ((x_list_t[i] - mu_x_t) @ (x_list_tminus1[i] - mu_x_tminus1).T)

        return V_collapsed_t_tminus1


    @staticmethod
    def Collapse(components: list, weights: list, transforms: list):
        n_components = len(components)
        if n_components < 2:
            print('There are less than 2 components. Not necessary for collapsing.')
            return components[0]

        x = 0.0
        V = 0.0

        for n in range(n_components):
            x += weights[n] * transforms[n] @ components[n].mean

        for n in range(n_components):
            diff = transforms[n] @ components[n].mean - x
            V += weights[n] * (transforms[n] @ components[n].covar @ transforms[n].T + diff @ diff.T)

        return Gaussian(mean=x, covar=V)
