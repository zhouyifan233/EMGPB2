# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GaussianSequence
from bayou.models import LinearModel, ConstantVelocity
from bayou.expmax.lineargaussian import LinearGaussian


def test_em():
    """ """
    dataset = []
    for i in range(1, 3001):
        measurement = np.loadtxt("E:\\WPAFB-learn-dynamics\\grayscale\\track_%04d.csv" % i, delimiter=',')
        for_init_state = measurement[0,:].tolist()
        #for_init_state.extend([0, 0])
        initial_state = Gaussian(np.array(for_init_state).reshape(10,1), 5*np.eye(10))
        
        initial_model = LinearModel(np.eye(10), 5*np.eye(10), np.eye(10), 5*np.eye(10))
        measurement = np.expand_dims(measurement, axis=-1)
        sequence = GaussianSequence(measurement, initial_state)
        dataset.append(sequence)
        if np.mod(i, 10) == 0:
            print("File %d loaded ..." % i)

    model, dataset, LLs = LinearGaussian.EM(dataset, initial_model,
                                            max_iters=100, threshold=0.00001,
                                            learn_H=False, learn_R=True,
                                            learn_A=True, learn_Q=True, learn_init_state=False,
                                            keep_Q_structure=False, diagonal_Q=False)

    print("A: ")
    printarrayA = model.A
    printarrayA = np.array2string(printarrayA, max_line_width=1000)
    print(printarrayA)
    #print(model.A)
    print("Q: ")
    printarrayQ = model.Q
    printarrayQ = np.array2string(printarrayQ, max_line_width=1000)
    print(printarrayQ)
    #print(model.Q)
    print("H: ")
    printarrayH = model.H
    printarrayH = np.array2string(printarrayH, max_line_width=1000)
    print(printarrayH)
    #print(model.H)
    print("R: ")
    printarrayR = model.R
    printarrayR = np.array2string(printarrayR, max_line_width=1000)
    print(printarrayR)
    #print(model.R)
    print("List of log-likelihood: ")
    print(LLs)
    #print("initial state mean: ")
    #print(dataset[0].initial_state.mean)
    #print("initial state covariance: ")
    #print(dataset[0].initial_state.covar)

    filedecriptor = open("E:\\WPAFB-learn-dynamics\\result1.txt", "w")
    filedecriptor.write("A: \n")
    filedecriptor.write(printarrayA)
    filedecriptor.write("\n ------------------------------------------ \n")
    filedecriptor.write("Q: \n")
    filedecriptor.write(printarrayQ)
    filedecriptor.write("\n ------------------------------------------ \n")
    filedecriptor.write("H: \n")
    filedecriptor.write(printarrayH)
    filedecriptor.write("\n ------------------------------------------ \n")
    filedecriptor.write("R: \n")
    filedecriptor.write(printarrayR)
    filedecriptor.close()

test_em()
