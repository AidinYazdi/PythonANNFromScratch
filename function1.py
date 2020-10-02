# this function is an OR gate

# import required libraries
import numpy as np

# the inputs (x) and desired outputs (y)
function1_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
function1_desired_outputs = np.array([[0], [1], [1], [1]])

# save the inputs and desired outputs to .csv files
np.savetxt('function1/function1_inputs.csv', function1_inputs, delimiter=',')
np.savetxt('function1/function1_desired_outputs.csv', function1_desired_outputs, delimiter=',')

# the configuration of the ANN
input_nodes = 2
hidden1_nodes = 3
hidden2_nodes = 3
output_nodes = 1

# how the input and output arrays should be formatted
# (ann.py will use this information to correctly format the numpy arrays)
input_array_dim = 2
input_array_r = 4
input_array_c = 2
output_array_dim = 1
output_array_r = 4
output_array_c = 1

# whether or not the input and output data should be normalized/denormalized
input_normalization_min = 0
input_normalization_max = 1
output_normalization_min = 0
output_normalization_max = 1
