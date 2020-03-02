from simulation.generate_path_templates import *


# Create a testing track with one constant velocity model
measurement1 = create_path_constant_volocity_one_model(output_measurements='data/measurement1.csv', output_groundtruth='data/groundtruth1.csv')

# Create a testing track with two constant velocity models
measurement2 = create_path_constant_volocity_multi_model(output_measurements='data/measurement2.csv', output_groundtruth='data/groundtruth2.csv')

# Create a testing track with two random walk models
measurement3 = create_path_random_walk_multi_model(output_measurements='data/measurement3.csv', output_groundtruth='data/groundtruth3.csv')

