# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GaussianSequence
from bayou.models import LinearModel, ConstantVelocity
from bayou.expmax.lineargaussian import LinearGaussian


def test_em():
    F = np.asarray([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    H = np.asarray([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    """
    Gaussian(mean, covariance)
    LinearModel(A, Q, H, R)
    """
    initial_state = Gaussian(np.zeros([4, 1]), 10.0*np.eye(4))
    initial_model = LinearModel(F, 1.0*np.eye(4), H, 1.0*np.eye(2))
    #initial_model = ConstantVelocity(dt=1.0, q=1.0, r=1.0, state_dim=4, obs_dim=2)

    measurements = np.loadtxt('data/measurement1.csv', delimiter=',')
    measurements = np.expand_dims(measurements, axis=-1)
    sequence = GaussianSequence(measurements, initial_state)
    dataset = [sequence]

    model, dataset, LLs = LinearGaussian.EM(dataset, initial_model,
                                            max_iters=100, threshold=1e-8,
                                            learn_H=True, learn_R=True,
                                            learn_A=True, learn_Q=True, learn_init_state=True,
                                            keep_Q_structure=False, diagonal_Q=False)

    print('F: ')
    print(model.A)
    print('Q: ')
    print(model.Q)
    print('H: ')
    print(model.H)
    print('R: ')
    print(model.R)
    print('LLs: ')
    print(LLs)
    print('Init_state: ')
    print(dataset[0].initial_state.mean)
    #print(dataset[0].initial_state.covar)

    return model, dataset


model, dataset = test_em()
