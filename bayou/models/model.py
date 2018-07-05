# -*- coding: utf-8 -*-
import numpy as np


class Model():
    """Base model"""


class LinearModel(Model):
    """Generic Linear model"""

    def __init__(self, A, Q, H, R):
        self.A = A
        self.Q = Q
        self.H = H
        self.R = R


class ConstantVelocity(LinearModel):
    """Constant Velocity model

    Attributes
    ----------
    state_dim : int
    obs_dim : int
    A : np.ndarray
        (state_dim, state_dim)
    Q : np.ndarray
        (state_dim, state_dim)
    H : np.ndarray
        (obs_dim, state_dim)
    R : np.ndarray
        (obs_dim, obs_dim)
    """

    def __init__(self, dt=1.0, q=1.0, r=1.0, state_dim=2, obs_dim=1):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        dt2 = dt**2 / 2.0
        dt3 = dt**3 / 3.0
        n_dim = int(state_dim / 2)
        I = np.eye(n_dim)

        self.A = np.kron(np.asarray([
            [1.0, dt],
            [0., 1.0]
        ]), I)

        self.Q = q * np.kron(np.asarray([
            [dt3, dt2],
            [dt2, dt]
        ]), I)

        self.H = np.eye(state_dim)[:obs_dim]

        self.R = r * np.eye(obs_dim)

    def get_Q_structure(self):
        return self.Q / self.Q[-1, -1]


class BrownianMotion(LinearModel):
    """Brownian Motion model

    Attributes
    ----------
    state_dim : int
    obs_dim : int
    A : np.ndarray
        (state_dim, state_dim)
    Q : np.ndarray
        (state_dim, state_dim)
    H : np.ndarray
        (obs_dim, state_dim)
    R : np.ndarray
        (obs_dim, obs_dim)
    """

    def __init__(self, dt=1.0, q=1.0, r=1.0, state_dim=2, obs_dim=2):
        self.state_dim = 2
        self.obs_dim = 2
        I = np.eye(state_dim)
        self.A = np.eye(state_dim)
        self.Q = q * dt * np.eye(state_dim)
        self.H = np.eye(state_dim)[:obs_dim]
        self.R = r * np.eye(obs_dim)

    def get_Q_structure(self):
        return self.Q / self.Q[-1, -1]
