import numpy as np

from emgpb2.states import Gaussian, GaussianSequence, GMM, GMMSequence
from emgpb2.models import LinearModel, ConstantVelocity, RandomWalk
from emgpb2.EM import SKFEstimator
from emgpb2.EM import LinearGaussianEstimator


# The EM for one Linear Gaussian model.
# Estimate parameters of one Kalman filter.
# Constant Velocity Model.
def test_lg_cv_estimator(init_P=5.0, q=0.5, r=1.0,
                         state_dim=4, obs_dim=2,
                         input_measurement='data/measurement1.csv'):
    initial_state = Gaussian(np.zeros([state_dim, 1]), (init_P ** 2) * np.eye(state_dim))
    initial_model = ConstantVelocity(dt=1.0, q=q, r=r, state_dim=state_dim, obs_dim=obs_dim)
    if isinstance(input_measurement, str):
        measurements = np.loadtxt(input_measurement, delimiter=',')
    else:
        measurements = input_measurement
    if measurements.ndim == 2:
        measurements = np.expand_dims(measurements, axis=-1)
    sequence = GaussianSequence(measurements, initial_state)
    dataset = [sequence]

    model, dataset, LLs = LinearGaussianEstimator.EM(dataset, initial_model,
                                                     max_iters=300, threshold=1e-6,
                                                     learn_H=True, learn_R=True,
                                                     learn_A=True, learn_Q=True, learn_init_state=True,
                                                     keep_Q_structure=False, diagonal_Q=False)

    return model, dataset


# The EM for GPB2.
# Estimate parameters of two Kalman filters.
# Two different Constant Velocity Models.
def test_skf_cv_estimator(init_P: list = [5.0, 5.0], q: list = [2.0, 10.0], r: list = [1.0, 1.0],
                          state_dim=4, obs_dim=2,
                          input_measurement='data/measurement2.csv'):
    """ 

    """
    # read measurement data
    if isinstance(input_measurement, str):
        measurements = np.loadtxt(input_measurement, delimiter=',')
    else:
        measurements = input_measurement
    if measurements.ndim == 2:
        measurements = np.expand_dims(measurements, axis=-1)
    # Initial state of measurements
    num_of_models = len(q)
    gaussian_models = []
    for i in range(num_of_models):
        gaussian_models.append(Gaussian(np.zeros([state_dim, 1]), (init_P[i] ** 2) * np.eye(state_dim)))
    initial_gmm_state = GMM(gaussian_models)
    # measurement sequence
    gmmsequence = GMMSequence(measurements, initial_gmm_state)
    dataset = [gmmsequence]
    # Initial models
    constantvelocity_models = []
    for i in range(num_of_models):
        constantvelocity_models.append(ConstantVelocity(dt=1.0, q=q[i], r=r[i], state_dim=state_dim, obs_dim=obs_dim))
    # Switching matrix
    Z = np.ones((2, 2)) / 2

    models_all, Z_all, dataset, LL = SKFEstimator.EM(dataset, constantvelocity_models, Z,
                                                     max_iters=300, threshold=1e-6, learn_H=True, learn_R=True,
                                                     learn_A=True, learn_Q=True, learn_init_state=False, learn_Z=True,
                                                     diagonal_Q=False, wishart_prior=False)

    return models_all, Z_all


# The EM for GPB2.
# Estimate parameters of two Kalman filters.
# Two different Random Walk Models.
def test_skf_rw_estimator(init_P: list = [5.0, 5.0], q: list = [1.0, 20.0], r: list = [2.0, 2.0],
                          state_dim=2, input_measurement='data/measurement3.csv'):
    """

    """
    # read measurement data
    if isinstance(input_measurement, str):
        measurements = np.loadtxt(input_measurement, delimiter=',')
    else:
        measurements = input_measurement
    if measurements.ndim == 2:
        measurements = np.expand_dims(measurements, axis=-1)
    # Initial state of measurements
    num_of_models = len(q)
    gaussian_models = []
    for i in range(num_of_models):
        gaussian_models.append(Gaussian(np.zeros([state_dim, 1]), (init_P[i] ** 2) * np.eye(state_dim)))
    initial_gmm_state = GMM(gaussian_models)
    # measurement sequence
    gmmsequence = GMMSequence(measurements, initial_gmm_state)
    dataset = [gmmsequence]
    # Initial models
    constantvelocity_models = []
    for i in range(num_of_models):
        constantvelocity_models.append(RandomWalk(q=q[i], r=r[i], state_dim=state_dim))
    # Switching matrix
    Z = np.ones((2, 2)) / 2

    models_all, Z_all, dataset, LL = SKFEstimator.EM(dataset, constantvelocity_models, Z,
                                                     max_iters=300, threshold=1e-6, learn_H=True, learn_R=True,
                                                     learn_A=True, learn_Q=True, learn_init_state=False, learn_Z=True,
                                                     diagonal_Q=False, wishart_prior=False)

    return models_all, Z_all
