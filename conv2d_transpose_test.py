
import tensorflow as tf
import numpy as np


def weight(name, shape):
	return tf.get_variable(name, shape, initializer= tf.contrib.layers.xavier_initializer(), dtype=tf.float64 )


def conv2d_transpose(X, kernel=[3,3,128,1], strides=[1,2,2,1], name="Upsample_1"):
	#SHAPE- tutorial
	### Conv ###
	#input = [64,7,7,3]
	# w = [3,3,3,256]
	# strides = [1,2,2,1] & SAME padding
	## outshape = [64,7,7,256]

	### Conv_Transpose ###
	#input = [64,7,7,3]
	# w = [3,3,128,3]
	# strides = [1,2,2,1] & SAME padding
	## outshape = [64,14,14,128]

	#Do not use input placeholder with NONE in batchsize as tf.nn.conv2d_transpose doesnt support automatic outshape interpretation
	with tf.variable_scope(name):			
		s = X.get_shape()
		output_shape = [s[0], s[1]*strides[1], s[2]*strides[2], kernel[2] ]
		
		W_convt = weight("W_convt", kernel)
		out = tf.nn.conv2d_transpose(X, W_convt, output_shape, strides, padding="SAME")
	return out





batch_size = 1

x = tf.placeholder(dtype=tf.float64, shape=(batch_size, 28,28,1))
kernel = [3,3,128,1]
strides = [1,2,2,1]

out = conv2d_transpose(x, kernel, strides, "Upsample")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print("\n\n#######################################################")
	
    xin = np.random.rand(batch_size,28,28,1)
    
    out = sess.run(out, feed_dict={x:xin})
    
    print("In- ", xin.shape)
    print("out- ", out.shape)
    
    
