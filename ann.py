# import required libraries
import numpy as np

# SETTINGS START:

# how many nodes in each layer
nodes_in_input_layer = 784
nodes_in_hidden_layer1 = 20
nodes_in_hidden_layer2 = 20
nodes_in_output_layer = 10

# whether or not the ANN should read in data
read_in_data = True

# whether or not the ANN should train
train = True

# whether or not the ANN should save data
save = True

# whether or not the ANN should display the weights and biases
display = False

# whether or not the ANN should test itself
test = True

# how many epochs the program should do
num_epochs = 1

# how often the program should tell the user its progress
# (measured in epochs)
progress_rate = 2000

# the learning rate
lr = 20

# how much the ANN will limit the range of the biases
bias_limiter = 1

# settings to do with the batches
batch_number = 1
batch_size = 1
batches_to_train = 60000 / batch_size
# batches_to_train = 5

# settings to do with testing
test_batch_number = 1
test_batch_size = 1
batches_to_test = 10000 / test_batch_size
# batches_to_test = 10

# SETTINGS END


# the function for getting the input and desired output features from the .csv file
# file should be either "mnist_train.csv" or "mnist_test.csv"
def get_inputs_and_outputs(file_name, temp_batch_number, temp_batch_size):
    input_arr = np.loadtxt(file_name, delimiter=',', ndmin=2, skiprows=((temp_batch_number - 1) * temp_batch_size), max_rows=temp_batch_size)
    # take the first element of each sub-array for the inputs (which is the label - the number
    # the ann is trying to approximate) and copy it into the target_outputs
    target_output_arr = np.zeros((batch_size, 10))
    for t in range(len(input_arr)):
        temp = input_arr[t][0]
        target_output_arr[t][int(temp)] = 1
    # delete the first element of each numpy sub-array (which is the label)
    input_arr = np.delete(input_arr, 0, axis=1)
    # reshape the inputs and outputs to make them compatible with the ann
    input_arr = input_arr.reshape(batch_size, 784)
    target_output_arr = target_output_arr.reshape(batch_size, 10)
    # return the arrays
    return (input_arr / 255), target_output_arr, (temp_batch_number + 1)


# actually setting up the inputs features and desired outputs for the first time
input_features, target_output, batch_number = get_inputs_and_outputs("mnist_train.csv", batch_number, batch_size)
# setting up the weights and biases
# reading in the weights and biases
if read_in_data:
    dim = 0
    if (nodes_in_input_layer * nodes_in_hidden_layer1) > 1:
        dim = 2
    else:
        dim = 1
    weight_hidden1 = np.loadtxt('weight_hidden1.csv', delimiter=',', ndmin=dim, dtype=np.float64)
    if (nodes_in_hidden_layer1 * nodes_in_hidden_layer2) > 1:
        dim = 2
    else:
        dim = 1
    weight_hidden2 = np.loadtxt('weight_hidden2.csv', delimiter=',', ndmin=dim, dtype=np.float64)
    if (nodes_in_hidden_layer2 * nodes_in_output_layer) > 1:
        dim = 2
    else:
        dim = 1
    weight_output = np.loadtxt('weight_output.csv', delimiter=',', ndmin=dim, dtype=np.float64)
    bias1 = np.loadtxt('bias1.csv', delimiter=',', ndmin=1, dtype=np.float64)
    bias2 = np.loadtxt('bias2.csv', delimiter=',', ndmin=1, dtype=np.float64)
    bias_op = np.loadtxt('bias_op.csv', delimiter=',', ndmin=1, dtype=np.float64)
    print("the data has been read in")
else:
    # randomly assigning the weights and biases if we are not
    # supposed to read them in from the .csv files
    weight_hidden1 = np.random.rand(nodes_in_input_layer, nodes_in_hidden_layer1)
    weight_hidden2 = np.random.rand(nodes_in_hidden_layer1, nodes_in_hidden_layer2)
    weight_output = np.random.rand(nodes_in_hidden_layer2, nodes_in_output_layer)
    bias1 = np.random.rand(nodes_in_hidden_layer1)
    bias2 = np.random.rand(nodes_in_hidden_layer2)
    bias_op = np.random.rand(nodes_in_output_layer)


# the sigmoid function
def sigmoid(x):
    # print(f"x = {x}")
    for j in range(len(x)):
        for h in range(len(x[j])):
            temp = x[j][h]
            if temp >= 0:
                x[j][h] = 1 / (1 + np.exp(-temp))
            else:
                temp = np.exp(temp)
                x[j][h] = temp / (1 + temp)
    return x


# the derivative of the sigmoid function
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


# forward propagation
# this function assumes that the inputs have been normalized
def forward_prop(arr):
    result1 = np.dot(arr, weight_hidden1) + bias1
    result2 = sigmoid(result1)
    result3 = np.dot(result2, weight_hidden2) + bias2
    result4 = sigmoid(result3)
    result5 = np.dot(result4, weight_output) + bias_op
    return sigmoid(result5)


# this function will return the amount of times that the largest element in
# arr1 is the same as the largest element in arr2 and the amount
# that it's not
def compare(arr1, arr2):
    index_max1 = np.argmax(arr1)
    index_max2 = np.argmax(arr2)
    if index_max1 == index_max2:
        return True
    else:
        return False


# a function to limit the range of a 1D array
def limit_range(arr, limit):
    for w in range(len(arr)):
        if arr[w] > limit:
            arr[w] = limit
        elif arr[w] < -1 * limit:
            arr[w] = -1 * limit
    return arr


# # settings to do with the batches
# batch_number = 1
# batch_size = 100
# backpropagation
if train:
    for epoch in range(num_epochs):
        batch_number = 1
        for current_batch in range(int(batches_to_train)):
            if current_batch != 0:
                # actually setting up the inputs features and desired outputs each time
                input_features, target_output, batch_number = get_inputs_and_outputs("mnist_train.csv", batch_number, batch_size)
            # multiply the input for the ANN matrix with the weights between
            # the input and hidden layer1 matrices to get the input for the
            # hidden layer1
            input_hidden1 = np.dot(input_features, weight_hidden1) + bias1

            # get the output of the hidden layer1
            # (which is just the sigmoid of the input to the hidden layer1)
            output_hidden1 = sigmoid(input_hidden1)

            # multiply the output of the hidden layer1 (which is the input for
            # the hidden layer2) with the weights between the hidden layer1 and
            # hidden layer2 to get the input for the hidden layer2
            input_hidden2 = np.dot(output_hidden1, weight_hidden2) + bias2

            # get the output of the hidden layer2
            # (which is just the sigmoid of the input to the hidden layer2)
            output_hidden2 = sigmoid(input_hidden2)

            # multiply the output of the hidden layer2 (which is the input for
            # the output layer) with the weights between the hidden layer2 and
            # output layer to get the input for the output layer
            input_op = np.dot(output_hidden2, weight_output) + bias_op

            # get the output of the output layer
            # (which is just the sigmoid of the input to the output layer)
            output_op = sigmoid(input_op)

            # derivatives for the output layer
            derror_douto = output_op - target_output
            douto_dino = sigmoid_der(input_op)
            # the derivative that will be used to calculate gradient descent
            # for the bias and for the output layer
            # print(f"derror_douto = {derror_douto}")
            deriv_op = derror_douto * douto_dino
            dino_dwo = output_hidden2
            # put all the derivatives together for derror/dwo (which is needed
            # for gradient descent)
            # we have to transpose the dino/dwo to make it compatible with the
            # other matrices
            derror_dwo = np.dot(dino_dwo.T, deriv_op)

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
            # separating out calculations to make gradient descent for the
            # bias easier
            deriv2 = douth_dinh2 * derror_douth2
            # combine all the previous derivatives into derror/dwh2
            # we have to transpose the dinh/dwh2 to make it compatible with the
            # other matrix
            derror_dwh2 = np.dot(dinh_dwh2.T, deriv2)

            # derivatives for the hidden layer1
            # derror/dino2 and douto/douth2 have been defined above
            # dino/douth2 needs to be transposed to make it a compatible matrix
            # derror_dino1 = derror_dino2 * dino_douth2.T
            derror_dino1 = np.dot(derror_dino2, dino_douth2.T)
            dino_douth1 = weight_hidden2
            # combine the two previous derivatives into derror/douth2
            # we have to transpose the dino/douth1 to make it compatible with
            # the other matrix
            derror_douth1 = np.dot(derror_dino1, dino_douth1.T)
            douth_dinh1 = sigmoid_der(input_hidden1)
            dinh_dwh1 = input_features
            # separating out calculations to make gradient descent for the
            # bias easier
            deriv1 = douth_dinh1 * derror_douth1
            # combine all the previous derivatives into derror/dwh1
            # we have to transpose the dinh/dwh1 to make it compatible with the
            # other matrix
            derror_dwh1 = np.dot(dinh_dwh1.T, deriv1)

            # update the weights
            weight_output -= lr * derror_dwo
            weight_hidden2 -= lr * derror_dwh2
            weight_hidden1 -= lr * derror_dwh1

            # update the biases
            for i in deriv_op:
                bias_op -= lr * i
            for j in deriv2:
                bias2 -= lr * j
            for k in deriv1:
                bias1 -= lr * k

            # limit the range of the bias values
            limit_range(bias_op, bias_limiter * nodes_in_input_layer)
            limit_range(bias2, bias_limiter * nodes_in_hidden_layer1)
            limit_range(bias1, bias_limiter * nodes_in_hidden_layer2)

            # display training data
            if current_batch % progress_rate == 0:
                print(f"current_batch = {current_batch}")
                print(f"epoch = {epoch}")
                # print(f"derror_dwo = {derror_dwo}")
                # print(f"derror_dwh2 = {derror_dwh2}")
                # print(f"derror_dwh1 = {derror_dwh1}")

# # save all the ANN data
if save:
    np.savetxt('weight_hidden1.csv', weight_hidden1, delimiter=',')
    np.savetxt('weight_hidden2.csv', weight_hidden2, delimiter=',')
    np.savetxt('weight_output.csv', weight_output, delimiter=',')
    np.savetxt('bias1.csv', bias1, delimiter=',')
    np.savetxt('bias2.csv', bias2, delimiter=',')
    np.savetxt('bias_op.csv', bias_op, delimiter=',')
    print("the ANN has been saved")

if display:
    # display the final value of the weights
    print("weight_hidden1")
    print(weight_hidden1)
    print("bias1")
    print(bias1)
    print("weight_hidden2")
    print(weight_hidden2)
    print("bias2")
    print(bias2)
    print("weight_output")
    print(weight_output)
    print("bias_op")
    print(bias_op)

# settings to do with testing
# test_batch_number = 1
# test_batch_size = 100
# batches_to_test = 1
if test:
    right = 0
    wrong = 0
    while test_batch_number < (batches_to_test + 1):
        test_inputs, test_desired_outputs, test_batch_number = get_inputs_and_outputs("mnist_test.csv", test_batch_number, test_batch_size)
        test_actual_outputs = forward_prop(test_inputs)
        for i in range(len(test_actual_outputs)):
            temp_test_actual_outputs = test_actual_outputs[i]
            temp_test_desired_outputs = test_desired_outputs[i]
            if compare(temp_test_actual_outputs, temp_test_desired_outputs):
                right += 1
            else:
                wrong += 1
            # print(f"temp_test_actual_outputs = {temp_test_actual_outputs}")
    print(f"right = {right}")
    print(f"wrong = {wrong}")
    print(f"percentage correct = {(right / (right + wrong)) * 100}%")
