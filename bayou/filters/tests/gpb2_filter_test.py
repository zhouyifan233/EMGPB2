# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GMM, GMMSequence
from bayou.models import LinearModel
from bayou.filters.skf import GPB2


def test_gpb2_filter():
    g1 = Gaussian(np.ones([2, 1]), np.eye(2))
    g2 = Gaussian(np.ones([4, 1]), np.eye(4))
    initial_gmm_state = GMM([g1, g2])

    measurements = 5 * np.random.randn(200, 2, 1) + 1

    gmmsequence = GMMSequence(measurements, initial_gmm_state)

    m1 = LinearModel(np.eye(2), np.eye(2), np.eye(2), np.eye(2))
    m2 = LinearModel(np.eye(4), np.eye(4), np.eye(4)[:2], np.eye(2))
    models = [m1, m2]

    Z = np.ones([2, 2]) / 2

    filtered_sequence = GPB2.filter_sequence(gmmsequence, models, Z)


test_gpb2_filter()
