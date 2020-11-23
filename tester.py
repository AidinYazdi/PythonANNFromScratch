# a tester file used for testing

import numpy as np

array1 = [1, 5, 3, 4]
array2 = [1, 2, 3, 4]


def compare(arr1, arr2):
    index_max1 = np.argmax(arr1)
    index_max2 = np.argmax(arr2)
    if index_max1 == index_max2:
        return True
    else:
        return False


temp = compare(array1, array2)
if temp:
    print("nice")
else:
    print("not nice")
