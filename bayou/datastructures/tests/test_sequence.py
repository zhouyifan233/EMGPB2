# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GaussianSequence


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


test_gaussiansequence()
