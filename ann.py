# import required libraries
import numpy as np

# SETTINGS START:

# Warning: batch_number, batch_size, test_batch_number, and test_batch_size must all be equal to 1.
# Otherwise, the program will break (these used to be configurable in the old way of loading in the datasets. In this
# version, that functionality has not yet been added in).

# how many nodes in each layer
nodes_in_input_layer = 784
nodes_in_hidden_layer1 = 20
nodes_in_hidden_layer2 = 20
nodes_in_output_layer = 10

# whether or not the ANN should read in data
read_in_data = False

# whether or not the ANN should train
train = True

# whether or not the ANN should save data
save = True

# whether or not the ANN should display the weights and biases
display = False

# whether or not the ANN should test itself
test = True

# how many epochs the program should do
num_epochs = 5

# how often the program should tell the user its progress
# (measured in epochs)
progress_rate = 2000

# the learning rate
lr = 0.01

# settings to do with biases
# how much the ANN will limit the range of the biases
bias_limiter = 4
bias_scalar = 0.001

# settings to do with the batches
batch_number = 1
batch_size = 5
# batches_to_train can go up to 60000
batches_to_train = 60000 / batch_size
# batches_to_train = 5
# to train the whole dataset, make this equal to 60000
elements_in_training_dataset = 60000

# settings to do with testing
test_batch_number = 1
# for now, test_batch_size has to be equal to batch_size. I might add functionality to change this in the future to
# make testing more efficient
test_batch_size = batch_size
# batches_to_test can go up to 10000
batches_to_test = 10000 / test_batch_size
# batches_to_test = 10
# to test the whole dataset, make this equal to e0000
elements_in_testing_dataset = 10000

# some checks to make sure that the settings are valid:
if (batches_to_train * batch_size) > elements_in_training_dataset:
    raise Exception("(batches_to_train * batch_size) must be less than elements_in_training_dataset")
if (batches_to_test * test_batch_size) > elements_in_testing_dataset:
    raise Exception("(batches_to_test * test_batch_size) must be less than elements_in_testing_dataset")
if (batches_to_train % batch_size) != 0:
    raise Exception("(batches_to_train % batch_size) must equal 0")
if (batches_to_test % test_batch_size) != 0:
    raise Exception("(batches_to_test % test_batch_size) must equal 0")

# SETTINGS END


# Notes on the new vs. old way of loading in training and testing data:
# The old way of loading in training and testing data to the ANN was to repeatedly access to file with each new data
# point. This involved reading an entire data point from a file 70,000 different times for each run of the entire
# training and testing datasets (and much more than one run is required to fully train the ANN). The advantage of that
# method is that it doesn't use a lot of RAM - RAM is only required to hold the weights and biases of the ANN itself and
# one data point (the single set of inputs and the single output) at a time. The drawback is that such a large amount of
# separate reads from a file is incredibly, incredibly slow (and also probably not so healthy for the drive). The new
# way of loading in this data involves loading the training and testing data to np arrays (which are held in RAM) at the
# beginning of the program and just accessing individual data points from there for the ANN to process one by one.
# Although this does involve holding an additional 69,999 data points in RAM at the same time, I think it's worth the
# incredibly speed increase and maintaining the health of the drive.

# loading in the entire training dataset (inputs and outputs)
if train:
    entire_training_dataset = np.loadtxt("mnist_train.csv", delimiter=',', ndmin=2, max_rows=elements_in_training_dataset)
    print("training dataset initialized")


# loading in the entire testing dataset (inputs and outputs)
if test:
    entire_testing_dataset = np.loadtxt("mnist_test.csv", delimiter=',', ndmin=2, max_rows=elements_in_testing_dataset)
    print("testing dataset initialized")


# the function for getting the input and desired output features from the .csv file
# file should be either "mnist_train.csv" or "mnist_test.csv"
def get_inputs_and_outputs(file_name, temp_batch_number, temp_batch_size):
    # make sure that temp_batch_size is equal to 1; functionality for larger batches has not been added in yet
    # if temp_batch_size != 1:
    #     raise Exception("temp_batch_size must be equal to 1. Functionality for larger batches has not been added in yet")

    # access the correct dataset and put the relevant part in input_arr
    # the reason it's a double array is to retain functionality with old code
    current_starting_index = (temp_batch_number - 1) * temp_batch_size
    if file_name == "mnist_train.csv":
        input_arr = np.array(entire_training_dataset[current_starting_index:(current_starting_index + temp_batch_size)])
    elif file_name == "mnist_test.csv":
        input_arr = np.array(entire_testing_dataset[current_starting_index:(current_starting_index + temp_batch_size)])
    else:
        raise Exception("You are trying to access a dataset that does not exist")

    # TESTING STUFF:
    # print(input_arr)

    # take the first element of each sub-array for the inputs (which is the label - the number
    # the ann is trying to approximate) and copy it into the target_outputs
    target_output_arr = np.zeros((batch_size, 10))
    for t in range(len(input_arr)):
        temp = input_arr[t][0]
        target_output_arr[t][int(temp)] = 1

    # delete the first element of input_arr (which is the label)
    if temp_batch_size == 1:
        input_arr = np.delete(input_arr, 0)
    else:
        input_arr = np.delete(input_arr, 0, axis=1)

    # TESTING STUFF:
    # print(input_arr)
    # print(target_output_arr)

    # reshape the inputs and outputs to make them compatible with the ann
    input_arr = input_arr.reshape(batch_size, 784)
    target_output_arr = target_output_arr.reshape(batch_size, 10)
    # return the arrays
    return (input_arr / 255), target_output_arr, (temp_batch_number + 1)


# actually setting up the inputs features and desired outputs for the first time
file_for_initialization = ""
if train:
    file_for_initialization = "mnist_train.csv"
elif test:
    file_for_initialization = "mnist_test.csv"
else:
    raise Exception("either train or test (at the beginning of the program) must be set to True")
input_features, target_output, batch_number = get_inputs_and_outputs(file_for_initialization, batch_number, batch_size)
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

            # # for testing purposes:
            # print(f'input_features = {input_features}')
            # print(f'input_features.shape = {input_features.shape}')
            # print(f'target_output = {target_output}')
            # print(f'target_output.shape = {target_output.shape}')

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
            # derror_douto = -(output_op - target_output)
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

            bias_op *= bias_scalar
            bias2 *= bias_scalar
            bias1 *= bias_scalar

            # limit the range of the bias values
            limit_range(bias_op, bias_limiter * nodes_in_input_layer)
            limit_range(bias2, bias_limiter * nodes_in_hidden_layer1)
            limit_range(bias1, bias_limiter * nodes_in_hidden_layer2)
            # limit_range(bias_op, 0)
            # limit_range(bias2, 0)
            # limit_range(bias1, 0)

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
        #TESTING STUFF:
        # print(f'test_batch_number = {test_batch_number}')
        # print(f'test_batch_size = {test_batch_size}')
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
