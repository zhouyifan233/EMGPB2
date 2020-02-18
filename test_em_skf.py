# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GMM, GMMSequence
from bayou.models import LinearModel, ConstantVelocity
from bayou.expmax.skf import SKF


def get_Q(Q_sig, dt=1):
    Q = (Q_sig ** 2) * np.asarray([
        [(1/3)*np.power(dt, 3), 0.0, (1/2)*np.power(dt, 2), 0.0],
        [0.0, (1/3)*np.power(dt, 3), 0.0, (1/2)*np.power(dt, 2)],
        [(1/2) * np.power(dt, 2), 0.0, dt, 0.0],
        [0.0, (1/2) * np.power(dt, 2), 0.0, dt]
    ])
    return Q


def test_em_skf_1():
    F = np.asarray([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    H = np.asanyarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    """ 
        
    """
    g1 = Gaussian(np.zeros([4, 1]), 9.0 * np.eye(4))
    g2 = Gaussian(np.zeros([4, 1]), 9.0 * np.eye(4))
    g3 = Gaussian(np.zeros([4, 1]), 9.0 * np.eye(4))
    initial_gmm_state = GMM([g1, g2, g3])

    # measurements = 5 * np.random.randn(200, 2, 1) + 1

    measurements = np.loadtxt('data/measurement2.csv', delimiter=',')
    measurements = np.expand_dims(measurements, axis=-1)

    gmmsequence = GMMSequence(measurements, initial_gmm_state)

    m1 = LinearModel(F, get_Q(1.0), H, (0.1 ** 2) * np.eye(2))
    m2 = LinearModel(F, get_Q(3.0), H, (0.1 ** 2) * np.eye(2))
    m3 = LinearModel(F, get_Q(5.5), H, (0.1 ** 2) * np.eye(2))
    initial_models = [m1, m2, m3]

    # Z = np.ones((3, 3)) / 3
    Z = np.array([[0.7, 0.15, 0.15],
                  [0.15, 0.7, 0.15],
                  [0.15, 0.15, 0.7]])

    dataset = [gmmsequence]

    models_all, Z_all, dataset, LL = SKF.EM(dataset, initial_models, Z,
                                        max_iters=100, threshold=0.00001, learn_H=True, learn_R=True,
                                        learn_A=False, learn_Q=True, learn_init_state=False, learn_Z=True,
                                        keep_Q_structure=False, diagonal_Q=False, wishart_prior=False)

    return models_all, Z_all


def test_em_skf_2():
    F = np.eye(3)
    H = np.eye(3)

    g1 = Gaussian(np.zeros([3, 1]), 10 * np.eye(3))
    g2 = Gaussian(np.zeros([3, 1]), 10 * np.eye(3))
    g3 = Gaussian(np.zeros([3, 1]), 10 * np.eye(3))
    initial_gmm_state = GMM([g1, g2, g3])

    # measurements = 5 * np.random.randn(200, 2, 1) + 1

    measurements = np.loadtxt(r'data/measurement3.csv', delimiter=',')
    measurements = np.expand_dims(measurements, axis=-1)

    gmmsequence = GMMSequence(measurements, initial_gmm_state)

    m1 = LinearModel(F, (1.0 ** 2) * np.eye(3), H, (0.5 ** 2) * np.eye(3))
    m2 = LinearModel(F, (4.0 ** 2) * np.eye(3), H, (0.5 ** 2) * np.eye(3))
    m3 = LinearModel(F, (6.0 ** 2) * np.eye(3), H, (0.5 ** 2) * np.eye(3))
    initial_models = [m1, m2, m3]

    Z = np.ones([3, 3]) / 3

    dataset = [gmmsequence]

    models_all, Z_all, dataset, LL = SKF.EM(dataset, initial_models, Z,
                                        max_iters=100, threshold=0.000001, learn_H=False, learn_R=True,
                                        learn_A=False, learn_Q=True, learn_init_state=False, learn_Z=True,
                                        keep_Q_structure=False, diagonal_Q=False, wishart_prior=False)


    return models_all, Z_all


def test_em_skf_3():
    g1 = Gaussian(np.ones([4, 1]), np.eye(4))
    g2 = Gaussian(np.ones([2, 1]), np.eye(2))
    initial_gmm_state = GMM([g1, g2])

    # measurements = 5 * np.random.randn(200, 2, 1) + 1

    measurements = np.loadtxt('data/measurements.csv', delimiter=',')
    measurements = np.expand_dims(measurements, axis=-1)

    gmmsequence = GMMSequence(measurements, initial_gmm_state)

    m1 = LinearModel(np.eye(4), np.eye(4), np.eye(4)[:2], np.eye(2))
    m2 = LinearModel(np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    initial_models = [m1, m2]

    Z = np.ones([2, 2]) / 2

    dataset = [gmmsequence]

    new_models, Z, dataset, LL = SKF.EM(dataset, initial_models, Z,
                                        max_iters=100, threshold=0.0001, learn_H=False, learn_R=True,
                                        learn_A=True, learn_Q=True, learn_init_state=True, learn_Z=True,
                                        keep_Q_structure=False, diagonal_Q=False, wishart_prior=False)

    print(LL)
    print(Z)
    print(new_models[0].R)

    return new_models

models_all, Z_all = test_em_skf_1()

