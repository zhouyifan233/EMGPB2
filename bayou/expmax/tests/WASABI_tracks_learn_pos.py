# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GaussianSequence
from bayou.models import LinearModel, ConstantVelocity
from bayou.expmax.lineargaussian import LinearGaussian


def test_em():
    """ """
    dataset = []
    for i in range(1, 201):
        measurement = np.loadtxt("E:\\WPAFB-learn-dynamics\\data_global_xy\\track_%04d.csv" % i, delimiter=',')
        for_init_state = measurement[0,:].tolist()
        for_init_state.extend([0, 0])
        initial_state = Gaussian(np.array(for_init_state).reshape(4,1), 10*np.eye(4))
        
        initial_model = ConstantVelocity(dt=1.0, q=10.0, r=10.0, state_dim=4, obs_dim=2)
        measurement = np.expand_dims(measurement, axis=-1)
        sequence = GaussianSequence(measurement, initial_state)
        dataset.append(sequence)
        if np.mod(i, 10) == 0:
            print("File %d loaded ..." % i)

    model, dataset, LLs = LinearGaussian.EM(dataset, initial_model,
                                            max_iters=100, threshold=0.00000001,
                                            learn_H=True, learn_R=True,
                                            learn_A=True, learn_Q=True, learn_init_state=False,
                                            keep_Q_structure=False, diagonal_Q=False)

    print("A: ")
    printarray = model.A
    printarray = np.array2string(printarray, max_line_width=1000)
    print(printarray)
    #print(model.A)
    print("Q: ")
    printarray = model.Q
    printarray = np.array2string(printarray, max_line_width=1000)
    print(printarray)
    #print(model.Q)
    print("H: ")
    printarray = model.H
    printarray = np.array2string(printarray, max_line_width=1000)
    print(printarray)
    #print(model.H)
    print("R: ")
    printarray = model.R
    printarray = np.array2string(printarray, max_line_width=1000)
    print(printarray)
    #print(model.R)
    print("List of log-likelihood: ")
    print(LLs)
    #print("initial state mean: ")
    #print(dataset[0].initial_state.mean)
    #print("initial state covariance: ")
    #print(dataset[0].initial_state.covar)

test_em()

