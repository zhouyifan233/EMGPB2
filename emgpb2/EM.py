# -*- coding: utf-8 -*-
import copy
import numpy as np
from scipy import linalg
from emgpb2.filters import KalmanFilter
from emgpb2.utils import Utility
from emgpb2.filters import GPB2Filter
from emgpb2.smoothers import GPB2Smoother
from emgpb2.smoothers import RTSSmoother


class LinearGaussianEstimator:
    """ """

    @staticmethod
    def e_step(sequence, model):
        sequence = KalmanFilter.filter_sequence(sequence, model)
        sequence = RTSSmoother.smooth_sequence(sequence, model)
        return sequence

    @staticmethod
    def m_step(dataset, model, initial_model,
               learn_H, learn_R, learn_A, learn_Q, learn_init_state,
               keep_Q_structure, diagonal_Q):

        N = len(dataset)

        data_cardinality = 0
        for n in range(N):
            data_cardinality += dataset[n].len

        old_H = model.H
        old_R = model.R
        old_A = model.A
        old_Q = model.Q

        # Update Model
        if learn_A:
            P_tminus1 = 0
            P_t_tminus1 = 0
            for n in range(N):
                sequence = dataset[n]
                smoothed_x = np.array([state.mean for state in sequence.smoothed])
                smoothed_V = np.array([state.covar for state in sequence.smoothed])
                smooth_VVs = sequence.smooth_crossvar
                for t in range(1, sequence.len):
                    P_t_tminus1 += smooth_VVs[t] + smoothed_x[t] @ smoothed_x[t - 1].T
                    P_tminus1 += smoothed_V[t - 1] + smoothed_x[t - 1] @ smoothed_x[t - 1].T
            new_A = P_t_tminus1 @ linalg.inv(P_tminus1)
            model.A = new_A

        if learn_Q:
            P_t = 0
            P_t_tminus1 = 0
            P_tminus1_t = 0
            P_tminus1 = 0
            for n in range(N):
                sequence = dataset[n]
                smoothed_x = np.array([state.mean for state in sequence.smoothed])
                smoothed_V = np.array([state.covar for state in sequence.smoothed])
                smooth_VVs = sequence.smooth_crossvar
                for t in range(1, sequence.len):
                    P_t_tminus1 += smooth_VVs[t] + smoothed_x[t] @ smoothed_x[t - 1].T
                    P_tminus1 += smoothed_V[t - 1] + smoothed_x[t - 1] @ smoothed_x[t - 1].T
                    P_t += smoothed_V[t] + smoothed_x[t] @ smoothed_x[t].T
                    P_tminus1_t += smooth_VVs[t] + smoothed_x[t - 1] @ smoothed_x[t].T
            new_A = P_t_tminus1 @ linalg.inv(P_tminus1)
            if learn_A:
                numerator = P_t - new_A @ P_t_tminus1.T
            else:
                numerator = P_t - old_A @ P_tminus1_t - P_t_tminus1 @ old_A.T + old_A @ P_tminus1 @ old_A.T
            denominator = sequence.len - 1

            if keep_Q_structure:
                structure = initial_model.get_Q_structure()
                inv_struct = linalg.inv(structure)
                q = np.trace(inv_struct @ numerator) / denominator
                new_Q = structure * q
            else:
                new_Q = numerator / denominator

            if diagonal_Q:
                new_Q = np.diag(np.diag(new_Q))
            model.Q = (new_Q + new_Q.T) / 2

        if learn_H:
            y_t_times_x_t = 0
            P_t = 0
            for n in range(N):
                sequence = dataset[n]
                smoothed_x = np.array([state.mean for state in sequence.smoothed])
                smoothed_V = np.array([state.covar for state in sequence.smoothed])
                for t in range(0, sequence.len):
                    y_t_times_x_t += sequence.measurements[t] @ smoothed_x[t].T
                    P_t += smoothed_V[t] + smoothed_x[t] @ smoothed_x[t].T
            new_H = y_t_times_x_t @ linalg.inv(P_t)
            model.H = new_H

        if learn_R:
            y_t_times_x_t = 0
            P_t = 0
            y_t_times_y_t = 0
            x_t_times_y_t = 0
            x_t_times_x_t = 0
            for n in range(N):
                sequence = dataset[n]
                smoothed_x = np.array([state.mean for state in sequence.smoothed])
                smoothed_V = np.array([state.covar for state in sequence.smoothed])
                for t in range(0, sequence.len):
                    y_t_times_x_t += sequence.measurements[t] @ smoothed_x[t].T
                    P_t += smoothed_V[t] + smoothed_x[t] @ smoothed_x[t].T
                    y_t_times_y_t += sequence.measurements[t] @ sequence.measurements[t].T
                    x_t_times_y_t += smoothed_x[t] @ sequence.measurements[t].T
                    x_t_times_x_t += smoothed_x[t] @ smoothed_x[t].T
            new_H = y_t_times_x_t @ linalg.inv(P_t)
            if learn_H:
                numerator = y_t_times_y_t - (new_H @ x_t_times_y_t)
            else:
                numerator = y_t_times_y_t - (old_H @ x_t_times_y_t) - (y_t_times_x_t @ old_H.T) + old_H @ P_t @ old_H.T
            denominator = sequence.len
            new_R = numerator/denominator
            model.R = (new_R + new_R.T) / 2

        return model

    @staticmethod
    def EM(dataset, initial_model, max_iters=20, threshold=0.0001,
           learn_H=True, learn_R=True, learn_A=True, learn_Q=True, learn_init_state=True,
           keep_Q_structure=False, diagonal_Q=False):
        """ Expectation-Maximization for a Linear Gaussian State-Space model.

        Parameters
        ----------
        dataset : list of GaussianSequence objects
            N iid sequences

        Returns
        -------
        LinearModel : LinearModel object
        dataset : list of sequences
        LLs : list of log likelihoods at each iteration
        """

        N = len(dataset)
        model = copy.deepcopy(initial_model)
        LLs = []

        for i in range(max_iters):

            # E-Step
            for n in range(N):
                dataset[n] = LinearGaussianEstimator.e_step(dataset[n], model)

            # Check convergence
            sequence_LLs = [np.sum(sequence.loglikelihood) for sequence in dataset]
            LLs.append(np.sum(sequence_LLs))
            print("Log-likelihood: " + str(np.sum(sequence_LLs)))
            if len(LLs) > 1:
                if Utility.check_lik_convergence(LLs[-1], LLs[-2], threshold):
                    print('iterations:', i)
                    return model, dataset, LLs

            # M-Step
            model = LinearGaussianEstimator.m_step(dataset, model, initial_model,
                                          learn_H, learn_R,
                                          learn_A, learn_Q, learn_init_state,
                                          keep_Q_structure, diagonal_Q)

            print("Estimated F: ")
            print(model.A)
            print("Estimated Q: ")
            print(model.Q)
            print("Estimated H: ")
            print(model.H)
            print("Estimated R: ")
            print(model.R)
            print("iteration %d ... LL %.5f" % (i, np.sum(sequence_LLs)))
            print("-----------------------------------------------------------------------")

        print('Converged. Iterations:', i)
        return model, dataset, LLs


class SKFEstimator:
    """ """

    @staticmethod
    def e_step(gmmsequence, models, Z):
        gmmsequence = GPB2Filter.filter_sequence(gmmsequence, models, Z)
        gmmsequence = GPB2Smoother.smooth_sequence(gmmsequence, models, Z)
        return gmmsequence

    @staticmethod
    def m_step(dataset, models, initial_models, Z,
               learn_H, learn_R, learn_A, learn_Q, learn_init_state, learn_Z,
               diagonal_Q, wishart_prior):

        N = len(dataset)
        data_cardinality = 0
        for n in range(N):
            data_cardinality += dataset[n].len
        n_models = len(models)

        old_H = [model_.H for model_ in models]
        old_R = [model_.R for model_ in models]
        old_A = [model_.A for model_ in models]
        old_Q = [model_.Q for model_ in models]

        # Update Model
        if learn_A:
            for m in range(n_models):
                for n in range(N):
                    sequence = dataset[n]
                    weights = sequence.get_smooth_weights()
                    weights = Utility.annealing_weights(weights)
                    x_t = np.array([state.mean for state in sequence.smoothed_collapsed])
                    V_t = np.array([state.covar for state in sequence.smoothed_collapsed])
                    V_t_tminus1 = sequence.smoothed_crossvar_collapsed
                    transforms_ = [s_.transforms for s_ in sequence.smoothed]
                    P_t_tminus1 = 0.0
                    P_tminus1 = 0.0
                    for t in range(1, sequence.len):
                        P_t_tminus1 += weights[t, m] * (V_t_tminus1[t] + x_t[t] @ x_t[t - 1].T)
                        P_tminus1 += weights[t, m] * (V_t[t-1] + x_t[t-1] @ x_t[t-1].T)
                    new_A = P_t_tminus1 @ linalg.inv(P_tminus1)
                    models[m].A = new_A

                print('model -- ' + str(m) + ' new_A: \n' + str(models[m].A))

        if learn_Q:
            for m in range(n_models):
                P_t = 0.0
                P_tminus1 = 0.0
                P_tminus1_t = 0.0
                P_t_tminus1 = 0.0
                W_sum = 0.0
                for n in range(N):
                    sequence = dataset[n]
                    weights = sequence.get_smooth_weights()
                    weights = Utility.annealing_weights(weights)
                    x_t = np.array([state.mean for state in sequence.smoothed_collapsed])
                    V_t = np.array([state.covar for state in sequence.smoothed_collapsed])
                    V_t_tminus1 = sequence.smoothed_crossvar_collapsed
                    transforms_ = [s_.transforms for s_ in sequence.smoothed]
                    for t in range(1, sequence.len):
                        P_t += weights[t, m] * (V_t[t] + x_t[t] @ x_t[t].T)
                        P_tminus1 += weights[t, m] * (V_t[t-1] + x_t[t-1] @ x_t[t-1].T)
                        P_tminus1_t += weights[t, m] * (V_t_tminus1[t] + x_t[t-1] @ x_t[t].T)
                        P_t_tminus1 += weights[t, m] * (V_t_tminus1[t] + x_t[t] @ x_t[t-1].T)
                        W_sum += weights[t, m]

                if wishart_prior:
                    alpha = 0.1 * data_cardinality
                    numerator = (
                        alpha * np.eye(old_Q[m].shape[0]) +
                        P_t - old_A[m] @ P_tminus1_t - P_t_tminus1 @ old_A[m].T + old_A[m] @ P_tminus1 @ old_A[m].T
                    )
                    denominator = (alpha + W_sum)
                else:
                    if learn_A:
                        numerator = P_t - models[m].A @ P_t_tminus1.T
                    else:
                        numerator = P_t - old_A[m] @ P_tminus1_t - P_t_tminus1 @ old_A[m].T + old_A[m] @ P_tminus1 @ old_A[m].T
                    denominator = W_sum
                new_Q = numerator / denominator

                if diagonal_Q:
                    new_Q = np.diag(np.diag(new_Q))
                # new_Q = (new_Q + new_Q.T)/2
                models[m].Q = new_Q
                print('model -- ' + str(m) + ' new_Q: \n' + str(models[m].Q))

        if learn_H:
            for m in range(n_models):
                y_t_times_x_t = 0.0
                P_t = 0.0
                for n in range(N):
                    sequence = dataset[n]
                    weights = sequence.get_smooth_weights()
                    weights = Utility.annealing_weights(weights)
                    x_t = np.array([state.mean for state in sequence.smoothed_collapsed])
                    V_t = np.array([state.covar for state in sequence.smoothed_collapsed])
                    transforms_ = [s_.transforms for s_ in sequence.smoothed]
                    for t in range(1, sequence.len):
                        y_t_times_x_t += weights[t, m] * (sequence.measurements[t] @ x_t[t].T)
                        P_t += weights[t, m] * (V_t[t] + x_t[t] @ x_t[t].T)
                new_H = y_t_times_x_t @ linalg.inv(P_t)
                models[m].H = new_H
                print('model -- ' + str(m) + ' new_H: \n' + str(models[m].H))

        if learn_R:
            for m in range(n_models):
                y_t_times_x_t = 0.0
                P_t = 0.0
                y_t_times_y_t = 0.0
                x_t_times_y_t = 0.0
                W_sum = 0.0
                for n in range(N):
                    sequence = dataset[n]
                    weights = sequence.get_smooth_weights()
                    weights = Utility.annealing_weights(weights)
                    x_t = np.array([state.mean for state in sequence.smoothed_collapsed])
                    V_t = np.array([state.covar for state in sequence.smoothed_collapsed])
                    transforms_ = [s_.transforms for s_ in sequence.smoothed]
                    for t in range(0, sequence.len):
                        y_t_times_y_t += weights[t, m] * sequence.measurements[t] @ sequence.measurements[t].T
                        x_t_times_y_t += weights[t, m] * x_t[t] @ sequence.measurements[t].T
                        y_t_times_x_t += weights[t, m] * sequence.measurements[t] @ x_t[t].T
                        P_t += weights[t, m] * (V_t[t] + x_t[t] @ x_t[t].T)
                        W_sum += weights[t, m]
                denominator = W_sum
                if learn_H:
                    numerator = y_t_times_y_t - (models[m].H @ x_t_times_y_t)
                else:
                    numerator = y_t_times_y_t - old_H[m] @ x_t_times_y_t - y_t_times_x_t @ old_H[m].T + old_H[m] @ P_t @ old_H[m].T
                new_R = numerator / denominator
                # new_R = (new_R + new_R.T)/2
                models[m].R = new_R
                print('model -- ' + str(m) + ' new_R: \n' + str(models[m].R))

        if learn_Z:
            z_numerator = 0
            z_denominator = 0
            for n in range(N):
                sequence = dataset[n]
                Pr_Stplus1_St_y1T = sequence.get_smoothed_Pr_Stplus1_St_y1T()
                Pr_Stplus1_St_y1T = Pr_Stplus1_St_y1T[1:]
                z_numerator += np.sum(Pr_Stplus1_St_y1T, axis=0)
                z_denominator += sequence.len
            new_Z = z_numerator / (z_denominator - 1)
            Z = new_Z
            print('new_Z: \n' + str(new_Z))

        if learn_init_state:
            for n in range(N):
                dataset[n].initial_state = dataset[n].smoothed_collapsed

        return models, Z

    @staticmethod
    def EM(dataset, initial_models, Z, max_iters=20, threshold=0.0001,
           learn_H=True, learn_R=True, learn_A=True, learn_Q=True, learn_init_state=True, learn_Z=True,
           diagonal_Q=False, wishart_prior=False):
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
        models_all = []
        Z_all = []
        i = 0
        for i in range(max_iters):
            # E-Step
            for n in range(N):
                dataset[n] = SKFEstimator.e_step(dataset[n], models, Z)
            # Check convergence
            sequence_LLs = [np.sum(sequence.measurement_likelihood) for sequence in dataset]
            print(np.sum(sequence_LLs))
            LLs.append(np.sum(sequence_LLs))
            if len(LLs) > 1:
                if Utility.check_lik_convergence(LLs[-1], LLs[-2], threshold):
                    print('iterations:', i)
                    return models_all, Z_all, dataset, LLs

            # M-Step
            models, Z = SKFEstimator.m_step(dataset, models, initial_models, Z,
                                   learn_H, learn_R, learn_A, learn_Q,
                                   learn_init_state, learn_Z,
                                   diagonal_Q, wishart_prior)
            models_all.append(models)
            Z_all.append(Z)
            print('-----------------------------------------------')
        print('Converged. Iterations:', i)

        return models_all, Z_all, dataset, LLs