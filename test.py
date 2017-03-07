import numpy
import random
import theano

x = [i for i in range(20)]
y = [[1,2,3,4], [3,4,5,6], [5,6,7,8], [7,8,9,2]]
random.shuffle(y)
print(y)