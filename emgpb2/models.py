# -*- coding: utf-8 -*-
import numpy as np


class Model():
    """Base model"""


class LinearModel(Model):
    """Generic Linear model
    Attributes:

    A : Transition matrix
    Q : Process noise matrix
    H : Measurement matrix
    R : Measurement noise matrix
    """

    def __init__(self, A: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray):
        self.A = A
        self.Q = Q
        self.H = H
        self.R = R


class RandomWalk(LinearModel):
    """Random Walk model"""
    def __init__(self, q=1.0, r=1.0, state_dim=1):
        self.state_dim = state_dim
        self.obs_dim = state_dim
        A = np.eye(state_dim)
        Q = (q ** 2) * np.eye(state_dim)
        H = np.eye(state_dim)
        R = (r ** 2) * np.eye(state_dim)

        super().__init__(A, Q, H, R)


class ConstantVelocity(LinearModel):
    """Constant Velocity model
    Attributes:

    dt: Invariant time interval
    q: Process noise multiplier
    r: Measurement noise multiplier
    state_dim : State dimension
    obs_dim : Measurement dimension
    """

    def __init__(self, dt=1.0, q=1.0, r=1.0, state_dim=2, obs_dim=1):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        dt2 = dt**2 / 2.0
        dt3 = dt**3 / 3.0
        n_dim = int(state_dim / 2)
        I = np.eye(n_dim)

        A = np.kron(np.asarray([
            [1.0, dt],
            [0., 1.0]
        ]), I)
        Q = (q ** 2) * np.kron(np.asarray([
            [dt3, dt2],
            [dt2, dt]
            ]), I)
        H = np.eye(state_dim)[:obs_dim]
        R = (r ** 2) * np.eye(obs_dim)

        super().__init__(A, Q, H, R)
