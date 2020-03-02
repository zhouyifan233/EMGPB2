from simulation.test_estimator_templates import test_lg_cv_estimator, test_skf_cv_estimator, test_skf_rw_estimator
from simulation.generate_path_templates import create_path_constant_volocity_one_model, create_path_constant_volocity_multi_model, create_path_random_walk_multi_model

# Create a testing track with two constant velocity models
measurements = create_path_constant_volocity_multi_model(q=[1.0, 6.0], r=[0.75, 0.5], t=200, change_pnt=[100],
                                                         state_dim=4, obs_dim=2, output_measurements=None, output_groundtruth=None)
test_skf_cv_estimator(init_P=[5.0, 5.0], q=[2.0, 10.0], r=[1.0, 1.0],
                      state_dim=4, obs_dim=2, input_measurement=measurements)

