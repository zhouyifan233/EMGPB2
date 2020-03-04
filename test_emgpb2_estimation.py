import numpy as np
from simulation.test_estimator_templates import test_lg_cv_estimator, test_skf_cv_estimator, test_skf_rw_estimator
from simulation.generate_path_templates import create_path_constant_volocity_one_model, create_path_constant_volocity_multi_model, create_path_random_walk_multi_model

# Create a testing track with two constant velocity models
print("Groundtruth parameters (CV): ")
measurements_cv = create_path_constant_volocity_multi_model(q=[1.0, 6.0], r=[0.75, 0.5], t=200, change_pnt=[100],
                                                            state_dim=4, obs_dim=2, output_measurements=None, output_groundtruth=None)
print("Estimated parameters (CV): ")
models_all_cv, LLs_cv = test_skf_cv_estimator(init_P=[5.0, 5.0], q=[2.0, 10.0], r=[1.0, 1.0],
                                              state_dim=4, obs_dim=2, input_measurement=measurements_cv, verbose=False)
max_ind_cv = np.argmax(np.array(LLs_cv))
model_cv = models_all_cv[max_ind_cv-1]
for i, model_ in enumerate(model_cv):
    print('F_' + str(i) + ':')
    print(model_.A)
    print('H_' + str(i) + ':')
    print(model_.H)
    print('Q_' + str(i) + ':')
    print(model_.Q)
    print('R_' + str(i) + ':')
    print(model_.R)


print("======================================================")
print("======================================================")
# Create a testing track with two random walk models
print("Groundtruth parameters (RW): ")
measurements_rw = create_path_random_walk_multi_model(q=[1.0, 6.0], r=[0.75, 0.5], t=200, change_pnt=[100],
                                                      state_dim=2, output_measurements=None, output_groundtruth=None)
print("Estimated parameters (RW): ")
models_all_rw, LLs_rw = test_skf_rw_estimator(init_P=[5.0, 5.0], q=[2.0, 10.0], r=[1.0, 1.0],
                                              state_dim=2, input_measurement=measurements_rw, verbose=False)
max_ind_rw = np.argmax(np.array(LLs_rw))
model_rw = models_all_rw[max_ind_rw-1]
for i, model_ in enumerate(model_rw):
    print('F_' + str(i) + ':')
    print(model_.A)
    print('H_' + str(i) + ':')
    print(model_.H)
    print('Q_' + str(i) + ':')
    print(model_.Q)
    print('R_' + str(i) + ':')
    print(model_.R)

