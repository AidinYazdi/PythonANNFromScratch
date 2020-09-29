# import required libraries
import numpy as np

# define input features (the inputs to the ANN, which
# is an OR gate in the case)
input_features = np.array([[0,0],[0,1],[1,0],[1,1]])
# print the shape of the input_features array for the user to see
# should display: (4, 2)
print (input_features.shape)
input_features

# define the desired outputs (target outputs)
target_output = np.array([[0,1,1,1]])
# reshape the target output array into a vector that
# will work with the inputs and the ANN
target_output = target_output.reshape(4,1)
# print the shape of the target_output array for the user to see
# should display: (4, 1)
# (4, 1) is a matrix which is compatible
# with our input matrix which is (4, 2)
print(target_output.shape)
target_output

# define the weights
# 6 for the hidden layer
# 3 for the output layer
# 9 in total
# the structure of the ANN is:
# 2 input nodes, 3 hidden nodes, one output node
# (and an input layer bias and hidden layer bias - but the biases
# don't need their weights defined here)
weight_hidden = np.array([[-3.82270773, 2.36324803, 3.13867107],[-3.79490945, 2.45183148, 3.18541383]])
weight_output = np.array([[-13.22689481],[1.74391783],[3.50426502]])

# the learning rate
lr = 0.05

# the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# the derivative of the sigmoid function
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

train = False

# backpropagation
if train:
    for epoch in range(200000):
        # multiply the input for the ANN matrix with the weights between the
        # input and hidden layers matrix to get the input for the hidden layer
        input_hidden = np.dot(input_features, weight_hidden)

        # get the output of the hidden layer
        # (which is just the sigmoid of the input to the hidden layer)
        output_hidden = sigmoid(input_hidden)

        # multiply the output of the hidden layer (which is the input for the
        # output layer) with the weights between the hidden and output layers
        # to get the input for the output layer
        input_op = np.dot(output_hidden, weight_output)

        # get the output of the output layer - the output of the whole ANN
        # (which is just the sigmoid of the input to the output layer)
        output_op = sigmoid(input_op)

        # calculate the mean squeared error and display it to the user along
        # with which epoch the program is on
        error_out = ((1 / 2) * (np.power((output_op - target_output), 2)))
        print("epoch = ")
        print(epoch)
        print("error = ")
        print(error_out.sum())

        # derivatives for the output layer
        derror_douto = output_op - target_output
        douto_dino = sigmoid_der(input_op)
        dino_dwo = output_hidden
        # put all the derivatives together for derror/dwo (which is needed for
        # gradient descent)
        # we have to transpose the dino/dwo to make it compatible with the
        # other matrices
        derror_dwo = np.dot(dino_dwo.T, derror_douto * douto_dino)

        # derivatives for the hidden layer
        # derror/douto and douto/dino have been defined above
        derror_dino = derror_douto * douto_dino
        dino_douth = weight_output
        # combine the two previous derivatives into derror/douth
        # we have to transpose the dino/douth to make it compatible with the
        # other matrix
        derror_douth = np.dot(derror_dino, dino_douth.T)
        douth_dinh = sigmoid_der(input_hidden)
        dinh_dwh = input_features
        # combine all the previous derivatives into derror/dwh
        # we have to transpose the dinh/dwh to make it compatible with the
        # other matrix
        derror_dwh = np.dot(dinh_dwh.T, douth_dinh * derror_douth)

        # update the weights
        weight_output -= lr * derror_dwo
        weight_hidden -= lr * derror_dwh

# display the final value of the weights
print(weight_hidden)
print(weight_output)

# predicting for (0,0)
# taking the inputs
single_point = np.array([0,0])
# first step
result1 = np.dot(single_point, weight_hidden)
# second step
result2 = sigmoid(result1)
# third step
result3 = np.dot(result2, weight_output)
# fourth step
result4 = sigmoid(result3)
# print the final result
print("inputs:")
print(single_point)
print("outputs:")
print(result4)

# predicting for (1,0)
# taking the inputs
single_point = np.array([1,0])
# first step
result1 = np.dot(single_point, weight_hidden)
# second step
result2 = sigmoid(result1)
# third step
result3 = np.dot(result2, weight_output)
# fourth step
result4 = sigmoid(result3)
# print the final result
print("inputs:")
print(single_point)
print("outputs:")
print(result4)

# predicting for (0,1)
# taking the inputs
single_point = np.array([0,1])
# first step
result1 = np.dot(single_point, weight_hidden)
# second step
result2 = sigmoid(result1)
# third step
result3 = np.dot(result2, weight_output)
# fourth step
result4 = sigmoid(result3)
# print the final result
print("inputs:")
print(single_point)
print("outputs:")
print(result4)

# predicting for (1,1)
# taking the inputs
single_point = np.array([1,1])
# first step
result1 = np.dot(single_point, weight_hidden)
# second step
result2 = sigmoid(result1)
# third step
result3 = np.dot(result2, weight_output)
# fourth step
result4 = sigmoid(result3)
# print the final result
print("inputs:")
print(single_point)
print("outputs:")
print(result4)
