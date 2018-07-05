# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian


def test_gaussian():
    """Gaussian test"""
    mean = np.ones([3, 1])
    covar = np.eye(3)

    state = Gaussian(mean, covar)

    assert(state.dim == mean.shape[0])
    assert(np.array_equal(mean, state.mean))
    assert(np.array_equal(covar, state.covar))


test_gaussian()
