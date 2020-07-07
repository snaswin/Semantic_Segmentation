import numpy as np
import tensorflow as tf

a0 = np.array( [[0,1,2], [1,1,1] ] )
b0 = np.random.random((2,3,3))
a0shape = list(a0.shape)
a0shape.append(1)
a0shape = tuple(a0shape)
a0 = np.reshape(a0, a0shape)


# ~ print(a0)
# ~ print(b0)

# ~ a1 = a0.reshape(-1,1)
# ~ b1 = b0.reshape(-1,3)
# ~ print(a1)
# ~ print(b1)

"""
a is label
b is prediction

a0: (2,3,1)
b0: (2,3,3)

Output:
a1: (6,1)
b1: (6,3)

"""
a = tf.convert_to_tensor( a0 )
b = tf.convert_to_tensor( b0 )

a1 = tf.reshape(a, shape=(-1,a.shape[-1]))
b1 = tf.reshape(b, shape=(-1,b.shape[-1]))


print(a.shape)
print(b.shape)


print(a1.shape)
print(b1.shape)
