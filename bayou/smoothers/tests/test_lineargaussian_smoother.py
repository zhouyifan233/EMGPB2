# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GaussianSequence
from bayou.models import LinearModel
from bayou.filters.lineargaussian import Kalman
from bayou.smoothers.lineargaussian import RTS


def test_lineargaussian_smoother():
    measurements = np.loadtxt('measurements.csv', delimiter=',')
    measurements = np.expand_dims(measurements, axis=-1)
    model = LinearModel(np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    initial_state = Gaussian(measurements[0], np.eye(2))
    sequence = GaussianSequence(measurements, initial_state)
    filtered = Kalman.filter_sequence(sequence, model)
    smoothed = RTS.smooth_sequence(sequence, model)

    print('FILT')
    print(filtered.filter_crossvar[-1])
    print('SMOOTH')
    print(smoothed.smooth_crossvar[-1])
    print(np.sum(smoothed.loglikelihood))


test_lineargaussian_smoother()
