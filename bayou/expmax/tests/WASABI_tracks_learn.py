# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GaussianSequence
from bayou.models import LinearModel, ConstantVelocity
from bayou.expmax.lineargaussian import LinearGaussian


def test_em():
    """ """
    dataset = []
    for i in range(1, 101):
        measurement = np.loadtxt("E:\\WPAFB-learn-dynamics\\data_global_xy\\track_%04d.csv" % i, delimiter=',')
        for_init_state = measurement[0,:].tolist()
        for_init_state.extend([0, 0])
        initial_state = Gaussian(np.array(for_init_state).reshape(4,1), 1*np.eye(4))
        
        initial_model = ConstantVelocity(1.0, 100.0, 4.0, 4, 2)
        measurement = np.expand_dims(measurement, axis=-1)
        sequence = GaussianSequence(measurement, initial_state)
        dataset.append(sequence)
        if np.mod(i, 10) == 0:
            print("File %d loaded ..." % i)

    model, dataset, LLs = LinearGaussian.EM(dataset, initial_model,
                                            max_iters=100, threshold=0.0001,
                                            learn_H=False, learn_R=False,
                                            learn_A=True, learn_Q=True, learn_init_state=False,
                                            keep_Q_structure=False, diagonal_Q=False)

    print("A: ")
    print(model.A)
    print("Q: ")
    print(model.Q)
    print("H: ")
    print(model.H)
    print("R: ")
    print(model.R)
    print("List of log-likelihood: ")
    print(LLs)
    #print("initial state mean: ")
    #print(dataset[0].initial_state.mean)
    #print("initial state covariance: ")
    #print(dataset[0].initial_state.covar)

test_em()
