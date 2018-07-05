# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GaussianSequence
from bayou.models import LinearModel, ConstantVelocity
from bayou.expmax.lineargaussian import LinearGaussian


def test_em():
    """ """
    initial_state = Gaussian(np.ones([2, 1]), np.eye(2))
    initial_model = LinearModel(np.eye(2), np.eye(2), np.eye(2), 0.01 * np.eye(2))

    measurements = np.loadtxt('C:\\Users\\yifan\\source\\repos\\bayou\\bayou\\expmax\\tests\\measurements.csv', delimiter=',')
    measurements = np.expand_dims(measurements, axis=-1)
    sequence = GaussianSequence(measurements, initial_state)
    dataset = [sequence]

    model, dataset, LLs = LinearGaussian.EM(dataset, initial_model,
                                            max_iters=100, threshold=0.000001,
                                            learn_H=True, learn_R=True,
                                            learn_A=True, learn_Q=True, learn_init_state=True,
                                            keep_Q_structure=False, diagonal_Q=False)

    print(model.A)
    print(model.Q)
    print(model.H)
    print(model.R)
    print(LLs)
    print(dataset[0].initial_state.mean)
    print(dataset[0].initial_state.covar)


test_em()
