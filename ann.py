# import required libraries
import numpy as np

# define input features (the inputs to the ANN, which
# is an OR gate in the case)
input_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# print the shape of the input_features array for the user to see
# should display: (4, 2)
# print (input_features.shape)
# input_features

# define the desired outputs (target outputs)
target_output = np.array([[0, 1, 1, 1]])
# reshape the target output array into a vector that
# will work with the inputs and the ANN
target_output = target_output.reshape(4, 1)
# print the shape of the target_output array for the user to see
# should display: (4, 1)
# (4, 1) is a matrix which is compatible
# with our input matrix which is (4, 2)
# print(target_output.shape)
# target_output

# define the weights
# 6 for the hidden layer
# 3 for the output layer
# 9 in total
# the structure of the ANN is:
# 2 input nodes, 3 hidden nodes, one output node
# (and an input layer bias and hidden layer bias - but the biases
# don't need their weights defined here)
weight_hidden1 = np.array([[-8.00442426, 7.59876266, 7.76072063], [-8.00390399, 7.59932299, 7.76118024]])
weight_hidden2 = np.array([[6.96980126, -2.97668445, -2.84868599], [-4.63805025, 3.77585312, 3.6671216],
                           [-5.43330553, 3.40792605, 3.78959919]])
weight_output = np.array([[-29.36733248], [1.70604094], [1.35045945]])

# the learning rate
lr = 0.025


# the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# the derivative of the sigmoid function
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


# forward propagation
def forwardProp(x, y):
    # predicting for (x,y)
    # taking the inputs
    single_point = np.array([x, y])
    # first step
    result1 = np.dot(single_point, weight_hidden1)
    # second step
    result2 = sigmoid(result1)
    # third step
    result3 = np.dot(result2, weight_hidden2)
    # fourth step
    result4 = sigmoid(result3)
    # fifth step
    result5 = np.dot(result4, weight_output)
    # sixth step
    result6 = sigmoid(result5)
    # print the final result
    print("inputs:")
    print(single_point)
    print("outputs:")
    print(result6)


train = True

# backpropagation
if train:
    for epoch in range(200000):
        # multiply the input for the ANN matrix with the weights between
        # the input and hidden layer1 matrices to get the input for the
        # hidden layer1
        input_hidden1 = np.dot(input_features, weight_hidden1)

        # get the output of the hidden layer1
        # (which is just the sigmoid of the input to the hidden layer1)
        output_hidden1 = sigmoid(input_hidden1)

        # multiply the output of the hidden layer1 (which is the input for
        # the hidden layer2) with the weights between the hidden layer1 and
        # hidden layer2 to get the input for the hidden layer2
        input_hidden2 = np.dot(output_hidden1, weight_hidden2)

        # get the output of the hidden layer2
        # (which is just the sigmoid of the input to the hidden layer2)
        output_hidden2 = sigmoid(input_hidden2)

        # multiply the output of the hidden layer2 (which is the input for
        # the output layer) with the weights between the hidden layer2 and
        # output layer to get the input for the output layer
        input_op = np.dot(output_hidden2, weight_output)

        # get the output of the output layer
        # (which is just the sigmoid of the input to the output layer)
        output_op = sigmoid(input_op)

        if epoch % 1000 == 0:
            # calculate the mean squeared error and display it to the user
            # along with which epoch the program is on
            error_out = ((1 / 2) * (np.power((output_op - target_output), 2)))
            print("epoch =")
            print(epoch)
            print("error =")
            print(error_out.sum())

        # derivatives for the output layer
        derror_douto = output_op - target_output
        douto_dino = sigmoid_der(input_op)
        dino_dwo = output_hidden2
        # put all the derivatives together for derror/dwo (which is needed
        # for gradient descent)
        # we have to transpose the dino/dwo to make it compatible with the
        # other matrices
        derror_dwo = np.dot(dino_dwo.T, derror_douto * douto_dino)

        # derivatives for the hidden layer2
        # derror/douto and douto/dino have been defined above
        derror_dino2 = derror_douto * douto_dino
        dino_douth2 = weight_output
        # combine the two previous derivatives into derror/douth1
        # we have to transpose the dino/douth2 to make it compatible with
        # the other matrix
        derror_douth2 = np.dot(derror_dino2, dino_douth2.T)
        douth_dinh2 = sigmoid_der(input_hidden2)
        dinh_dwh2 = input_hidden1
        # combine all the previous derivatives into derror/dwh2
        # we have to transpose the dinh/dwh2 to make it compatible with the
        # other matrix
        derror_dwh2 = np.dot(dinh_dwh2.T, douth_dinh2 * derror_douth2)

        # derivatives for the hidden layer1
        # derror/dino2 and douto/douth2 have been defined above
        # dino/douth2 needs to be transposed to make it a compatable matrix
        derror_dino1 = derror_dino2 * dino_douth2.T
        dino_douth1 = weight_hidden2
        # combine the two previous derivatives into derror/douth2
        # we have to transpose the dino/douth1 to make it compatible with
        # the other matrix
        derror_douth1 = np.dot(derror_dino1, dino_douth1.T)
        douth_dinh1 = sigmoid_der(input_hidden1)
        dinh_dwh1 = input_features
        # combine all the previous derivatives into derror/dwh1
        # we have to transpose the dinh/dwh1 to make it compatible with the
        # other matrix
        derror_dwh1 = np.dot(dinh_dwh1.T, douth_dinh1 * derror_douth1)

        # update the weights
        weight_output -= lr * derror_dwo
        weight_hidden2 -= lr * derror_dwh2
        weight_hidden1 -= lr * derror_dwh1

# display the final value of the weights
print(weight_hidden1)
print(weight_hidden2)
print(weight_output)

forwardProp(0, 0)
forwardProp(1, 0)
forwardProp(0, 1)
forwardProp(1, 1)