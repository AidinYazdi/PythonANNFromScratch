# this function is a sin function

# import required libraries
import numpy as np

# how the input and output data should be normalized/denormalized
input_normalization_min = 0
input_normalization_max = 1
output_normalization_min = 0
output_normalization_max = 1

# the number of inputs and outputs we want
number_of_inputs = 500
number_of_outputs = number_of_inputs

# the inputs and desired outputs
# setting up the numpy arrays
input_size = (number_of_inputs, 10)
function2_inputs = np.empty(input_size, dtype=np.float64)
function2_inputs.fill(0)
output_size = (number_of_outputs, 10)
function2_desired_outputs = np.empty(output_size, dtype=np.float64)
function2_desired_outputs.fill(0)
# doing the actual calculations (assign values to the arrays)
for i in range(len(function2_inputs)):
    function2_inputs[i, np.random.randint(10)] = 1.0

for i in range(len(function2_inputs)):
    for j in range(len(function2_inputs[i])):
        if function2_inputs[i, j] == 1:
            function2_desired_outputs[i, j] = 1

# save the inputs and desired outputs to .csv files
np.savetxt('function2/function2_inputs.csv', function2_inputs, delimiter=',')
np.savetxt('function2/function2_desired_outputs.csv', function2_desired_outputs, delimiter=',')

# the configuration of the ANN
input_nodes = 10
hidden1_nodes = 16
hidden2_nodes = 16
output_nodes = 10

# how the input and output arrays should be formatted
# (ann.py will use this information to correctly format the numpy arrays)
input_array_dim = 2
input_array_r = number_of_inputs
input_array_c = 10
output_array_dim = 1
output_array_r = number_of_outputs
output_array_c = 10
