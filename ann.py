# an ANN that will approximate an OR gate
# import required libraries
import numpy as np

# define import vectors (called input_features)
input_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# the shape is printed out in the form (r, c)
# where r is the rows in the matrix and c is the column
print(input_features.shape)
input_features

# define the desired output vectors (target_output)
target_output = np.array([[0, 1, 1, 1]])
# reshape the target_output into vectors
target_output = target_output.reshape(4, 1)
print(target_output.shape)
target_output

# define the weights vector
weights = np.array([[0.1], [0.2]])
print(weights.shape)
weights

# the bias weight
bias = 0.3

# the learning rate
lr = 0.05


# the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# the derivative of the sigmoid function
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


# backpropagation
# change the number in the for loop to change the number of epochs
for epoch in range(10000):
    # defining our input vectors for this epoch as the input vectors
    # defined above
    inputs = input_features

    # forwardpropagation
    # (so that we can get output values to use to find
    # error for backpropogation)

    # feedforward input
    # setting the input into the output neuron as the dot product of
    # the inputs vector and weights vector plus the bias
    # So: in_o = (i1 * w1) + (i2 * w2) + b
    in_o = np.dot(inputs, weights) + bias

    # feedforward output
    # this is the output of the output neuron (output = sigmoid(input))
    # remember: this ANN only has input and output layers,
    # no hidden layers, so this is the final output of the ANN
    out_o = sigmoid(in_o)

    # the actual backpropagation now that we have
    # the outputs from forwardpropagation

    # we need to always find the error because we will use it for backpropagation
    error = out_o - target_output
    if epoch % 1000 == 0:
        # show the user which epoch the ANN is up to
        print(epoch)
        # calculating error
        # the error will be used in gradient descent on a weight by weight basis
        # we add another variable here with the sum of the errors of all the
        # weights to display to the user to show the user
        # how well the ANN is doing
        x = error.sum()
        # display the error for the user
        print(x)

    # gradient descent

    # calculating the derivative
    # the error has been defined about as: error = out_o - target_output
    derror_douto = error
    douto_dino = sigmoid_der(out_o)

    # multiplying the derivatives
    # this is part of the gradient descent formula
    # check notebook for more details on the formula
    deriv = derror_douto * douto_dino

    # now, we must multiply the previous derivatives with din_o/dw
    # (which is also just the output value for the previous node connected
    # to the given weight, or the input value for the whole network since
    # this network only has input and output layers)
    # we have to get the transpose of input_features in order to organize
    # the matrix so that it can be multiplied with the
    # other derivative matrices
    inputs = input_features.T
    deriv_final = np.dot(inputs, deriv)

    # updating the values of the weights
    weights -= lr * deriv_final

    # updating the bias weight value
    # since the bias doesn't get an input
    # we don't need to factor in din_o/dw
    # remember, "i" will equal each number in
    # the deriv matrix throughout the for loop
    for i in deriv:
        bias -= lr * i

# check the final values for the weights and bias
print(weights)
print(bias)

# predicting the values of the OR gate

# predicting for (0,0)
# taking the inputs
single_point = np.array([0, 0])
# first step
result1 = np.dot(single_point, weights) + bias
# second step
result2 = sigmoid(result1)
# print the final result
print(result2)

# predicting for (1,0)
# taking the inputs
single_point = np.array([1, 0])
# first step
result1 = np.dot(single_point, weights) + bias
# second step
result2 = sigmoid(result1)
# print the final result
print(result2)

# predicting for (0,1)
# taking the inputs
single_point = np.array([0, 1])
# first step
result1 = np.dot(single_point, weights) + bias
# second step
result2 = sigmoid(result1)
# print the final result
print(result2)

# predicting for (1,1)
# taking the inputs
single_point = np.array([1, 1])
# first step
result1 = np.dot(single_point, weights) + bias
# second step
result2 = sigmoid(result1)
# print the final result
print(result2)
