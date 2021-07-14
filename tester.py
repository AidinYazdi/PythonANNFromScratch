import numpy as np

# the learning rate
# the starting learning rate
lr = 2
# the maximum and minimum learning rate
lr_max = 3
lr_min = 0.05
# an iterator to keep track of where we are on the learning rate iteration function (the update_lr function)
lr_iterator = 0
# how much lr_iterator should be iterated with each iteration
lr_iteration_rate = 0.1


# the function to iterate the learning rate (lr)
def update_lr(temp_iterator):
    new_lr = 0.5 * ((np.sin(temp_iterator) * (lr_max - lr_min)) + (lr_max + lr_min))
    temp_iterator += lr_iteration_rate
    return new_lr, temp_iterator


# update the learning rate
for i in range(300):
    print(f'lr = {lr}')
    lr, lr_iterator = update_lr(lr_iterator)
