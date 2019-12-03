# -*- coding: utf-8 -*-
import numpy as np

from bayou.datastructures import Gaussian, GMM


def test_gaussian():
    """Gaussian test"""
    mean = np.ones([3, 1])
    covar = np.eye(3)

    state = Gaussian(mean, covar)

    assert(state.dim == mean.shape[0])
    assert(np.array_equal(mean, state.mean))
    assert(np.array_equal(covar, state.covar))


def test_gmm():
    """Gaussian Mixture Model test"""

    g1 = Gaussian(np.ones([4, 1]), np.eye(4))
    g2 = Gaussian(np.ones([2, 1]) / 3, np.eye(2))
    components = [g1, g2]

    state = GMM(components)
    assert(state.n_components == len(components))
    assert(state.gaussian_dims[0] == g1.dim)
    assert(state.gaussian_dims[1] == g2.dim)
    assert(
        np.array_equal(
            state.weights,
            np.log(np.ones([len(components), 1]) / len(components))
        )
    )
    assert(
        np.array_equal(
            state.transforms[0, 1],
            np.eye(4)[:2]
        )
    )
    assert(
        np.allclose(
            state.collapse().covar,
            np.array([[1.11111111, 0.11111111, 0.16666667, 0.16666667],
                      [0.11111111, 1.11111111, 0.16666667, 0.16666667],
                      [0.16666667, 0.16666667, 0.75, 0.25],
                      [0.16666667, 0.16666667, 0.25, 0.75]])
        )
    )


test_gaussian()
test_gmm()
