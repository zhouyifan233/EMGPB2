# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GMM, GaussianSequence, GMMSequence


def test_gaussiansequence():
    """Gaussian Sequence test"""

    measurements = np.ones([200, 4, 1])
    initial_state = Gaussian(measurements[0], np.eye(4))
    sequence = GaussianSequence(measurements, initial_state)

    assert(sequence.len == measurements.shape[0])
    assert(sequence.filtered.shape[0] == sequence.len)
    assert(sequence.smoothed.shape[0] == sequence.len)
    assert(sequence.filtered.dtype == Gaussian)
    assert(sequence.smoothed.dtype == Gaussian)
    assert(
        np.array_equal(initial_state.mean,
                       np.ones([4, 1]))
    )


def test_gmmsequence():
    """GMM Sequence test"""

    measurements = np.ones([5, 4, 1])
    g1 = Gaussian(measurements[0], np.eye(4))
    g2 = Gaussian(measurements[0], np.eye(4))
    initial_state = GMM([g1, g2])
    sequence = GMMSequence(measurements, initial_state)

    assert(sequence.len == measurements.shape[0])
    assert(sequence.filtered.shape[0] == sequence.len)
    assert(sequence.smoothed.shape[0] == sequence.len)
    assert(sequence.filtered.dtype == Gaussian)
    assert(sequence.smoothed.dtype == Gaussian)
    assert(
        np.array_equal(initial_state.collapse().mean,
                       np.ones([4, 1]))
    )


test_gaussiansequence()
test_gmmsequence()
