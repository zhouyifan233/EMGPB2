# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GMM, GMMSequence
from bayou.models import LinearModel
from bayou.filters.skf import GPB2 as GPB2f
from bayou.smoothers.skf import GPB2 as GPB2s


def test_gpb2_smoother():
    g1 = Gaussian(np.ones([4, 1]), np.eye(4))
    g2 = Gaussian(np.ones([2, 1]), np.eye(2))
    initial_gmm_state = GMM([g1, g2])

    measurements = 5 * np.random.randn(200, 2, 1) + 1

    measurements = np.loadtxt(r'measurements.csv', delimiter=',')
    measurements = np.expand_dims(measurements, axis=-1)

    gmmsequence = GMMSequence(measurements, initial_gmm_state)

    m1 = LinearModel(np.eye(4), np.eye(4), np.eye(4)[:2], np.eye(2))
    m2 = LinearModel(np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    models = [m1, m2]

    Z = np.ones([2, 2]) / 2
    filtered_sequence = GPB2f.filter_sequence(gmmsequence, models, Z)
    smoothed_sequence = GPB2s.smooth_sequence(filtered_sequence, models, Z)

    fp = smoothed_sequence.get_filter_estimates()
    sp = smoothed_sequence.get_smooth_estimates()

    #print(smoothed_sequence.filter_crossvar[0,1][10])
    #print(np.sum(smoothed_sequence.measurement_likelihood))

    #print(smoothed_sequence.smoothed[-1])
    #print(smoothed_sequence.get_smooth_weights(1)[0])
    print(smoothed_sequence.smooth_joint_pr[30, 0, 1].shape)
    print(smoothed_sequence.smooth_crossvar[0, 1][10].shape)
    print(smoothed_sequence.smooth_j_k_t[123, 0, 0].mean)
    print(smoothed_sequence.smooth_joint_pr[20])


test_gpb2_smoother()
