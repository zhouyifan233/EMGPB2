# -*- coding: utf-8 -*-
import copy
import multiprocessing

import numpy as np
from scipy import linalg

from bayou.expmax.base import EM
#from bayou.datastructures import Gaussian, GMM
from bayou.filters.skf import GPB2 as GPB2f
from bayou.smoothers.skf import GPB2 as GPB2s

from bayou.utils.util import Utility


class SKF(EM):
    """ """

    @staticmethod
    def e_step(gmmsequence, models, Z):
        gmmsequence = GPB2f.filter_sequence(gmmsequence, models, Z)
        gmmsequence = GPB2s.smooth_sequence(gmmsequence, models, Z)
        return gmmsequence

    @staticmethod
    def m_step(dataset, models, initial_models, Z,
               learn_H, learn_R, learn_A, learn_Q, learn_init_state, learn_Z,
               keep_Q_structure, diagonal_Q, wishart_prior):

        N = len(dataset)

        data_cardinality = 0
        for n in range(N):
            data_cardinality += dataset[n].len

        n_models = len(models)

        H_terms = np.empty([n_models], dtype=list)
        R_terms = np.empty([n_models], dtype=list)
        A_terms = np.empty([n_models], dtype=list)
        Q_terms = np.empty([n_models], dtype=list)
        for n in range(n_models):
            H_terms[n] = [0, 0]
            R_terms[n] = [0, 0, 0]
            A_terms[n] = [0, 0]
            Q_terms[n] = [0, 0, 0]

        for n in range(N):
            sequence = dataset[n]

            for m in range(n_models):
                weights = np.exp(sequence.get_smooth_weights(m))
                x_T = np.array([state.components[m].mean for state in sequence.smoothed])
                V_T = np.array([state.components[m].covar for state in sequence.smoothed])

                if learn_H or learn_R:
                    term_0 = 0
                    term_1 = 0
                    for t in range(0, sequence.len):
                        term_0 += weights[t] * sequence.measurements[t] @ x_T[t].T
                        term_1 += weights[t] * (V_T[t] + x_T[t] @ x_T[t].T)
                    H_terms[m][0] += term_0
                    H_terms[m][1] += term_1

                if learn_R:
                    term_0 = 0
                    term_1 = 0
                    term_2 = 0
                    for t in range(0, sequence.len):
                        term_1 += weights[t] * sequence.measurements[t] @ sequence.measurements[t].T
                        term_2 += weights[t] * x_T[t] @ sequence.measurements[t].T
                        term_0 += weights[t]
                    R_terms[m][0] += term_0
                    R_terms[m][1] += term_1
                    R_terms[m][2] += term_2

                if learn_A or learn_Q:
                    term_0 = 0
                    term_1 = 0
                    for t in range(1, sequence.len):
                        for j in range(n_models):
                            term_0 += np.exp(sequence.smooth_joint_pr[t, j, m]) * (sequence.smooth_crossvar[j, m][t] + x_T[t] @ sequence.smooth_j_k_t[t - 1, j, m].mean.T)
                            term_1 += np.exp(sequence.smooth_joint_pr[t, j, m]) * (V_T[t - 1] + x_T[t - 1] @ x_T[t - 1].T)
                    A_terms[m][0] += term_0
                    A_terms[m][1] += term_1

                if learn_Q:
                    term_0 = 0
                    term_1 = 0
                    term_2 = 0
                    for t in range(1, sequence.len):
                        for j in range(n_models):
                            t1 = np.exp(sequence.smooth_joint_pr[t, j, m]) * (V_T[t] + x_T[t] @ x_T[t].T)
                            term_1 += t1
                            t2 = np.exp(sequence.smooth_joint_pr[t, j, m]) * (sequence.smooth_crossvar[j, m][t] + sequence.smooth_j_k_t[t - 1, j, m].mean @ x_T[t].T)
                            term_2 += t2

                            term_0 += np.exp(sequence.smooth_joint_pr[t, j, m])
                    Q_terms[m][0] += term_0
                    Q_terms[m][1] += term_1
                    Q_terms[m][2] += term_2

        # Update Model
        if learn_R:
            for m in range(n_models):
                new_R = (R_terms[m][1] - models[m].H @ R_terms[m][2] - (R_terms[m][2].T @ models[m].H.T) + models[m].H @ H_terms[m][1] @ models[m].H.T) / R_terms[m][0]
                models[m].R = new_R

        if learn_H:
            for m in range(n_models):
                new_H = H_terms[m][0] @ linalg.inv(H_terms[m][1])
                models[m].H = new_H

        if learn_Q:
            for m in range(n_models):
                if wishart_prior:
                    alpha = 0.1 * data_cardinality
                    numerator = (
                        alpha * np.eye(models[m].Q.shape[0]) +
                        Q_terms[m][1] - models[m].A @ Q_terms[m][2] - A_terms[m][0] @ models[m].A.T + models[m].A @ A_terms[m][1] @ models[m].A.T
                    )
                    denominator = (alpha + Q_terms[m][0])
                else:
                    numerator = Q_terms[m][1] - models[m].A @ Q_terms[m][2] - A_terms[m][0] @ models[m].A.T + models[m].A @ A_terms[m][1] @ models[m].A.T
                    denominator = Q_terms[m][0]

                if keep_Q_structure:
                    structure = initial_models[m].get_Q_structure()
                    inv_struct = linalg.inv(structure)
                    q = np.trace(inv_struct @ numerator) / denominator
                    new_Q = structure * q
                else:
                    new_Q = numerator / denominator

                if diagonal_Q:
                    new_Q = np.diag(np.diag(new_Q))
                models[m].Q = new_Q

        if learn_A:
            for m in range(n_models):
                new_A = A_terms[m][0] @ linalg.inv(A_terms[m][1])
                models[m].A = new_A

        z_numerator = 0
        z_denominator = 0
        for n in range(N):
            sequence = dataset[n]
            if learn_Z:
                sum_joint = np.sum(np.exp(sequence.smooth_joint_pr), axis=0)
                z_numerator += sum_joint
                z_denominator += np.sum(sum_joint, axis=1, keepdims=True)

            if learn_init_state:
                sequence.initial_state = sequence.smoothed[0]

        if learn_Z:
            new_Z = z_numerator / z_denominator
            Z = new_Z

        return models, Z

    @staticmethod
    def EM(dataset, initial_models, Z, max_iters=20, threshold=0.0001,
           learn_H=True, learn_R=True, learn_A=True, learn_Q=True, learn_init_state=True, learn_Z=True,
           keep_Q_structure=False, diagonal_Q=False, wishart_prior=False):
        """ Expectation-Maximization for a Linear Gaussian State-Space model.

        Parameters
        ----------
        dataset : list of GaussianSequence objects
            N iid sequences

        Returns
        -------
        list of models
        dataset : list of sequences
        LLs : list of log likelihoods at each iteration
        """

        N = len(dataset)
        models = copy.deepcopy(initial_models)
        LLs = []

        for i in range(max_iters):

            # E-Step
            for n in range(N):
                dataset[n] = SKF.e_step(dataset[n], models, Z)

            # Check convergence
            sequence_LLs = [np.sum(sequence.measurement_likelihood) for sequence in dataset]
            print(np.sum(sequence_LLs))
            LLs.append(np.sum(sequence_LLs))
            if len(LLs) > 1:
                if Utility.check_lik_convergence(LLs[-1], LLs[-2], threshold):
                    print('iterations:', i)
                    return models, Z, dataset, LLs

            # M-Step
            models, Z = SKF.m_step(dataset, models, initial_models, Z,
                                   learn_H, learn_R, learn_A, learn_Q,
                                   learn_init_state, learn_Z, keep_Q_structure,
                                   diagonal_Q, wishart_prior)

        print('Converged. Iterations:', i)
        return models, Z, dataset, LLs
