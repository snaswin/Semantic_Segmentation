import numpy as np
import tensorflow as tf
import pathlib
import math
import json
import glob
import matplotlib.pyplot as plt
#from sklearn.utils import shuffle as sk_shuffle
np.random.seed(4)
from copy import deepcopy
import cv2
from tqdm import tqdm
import os


#parameter utils
def weight(name, shape):
	return tf.get_variable(name, shape, initializer= tf.contrib.layers.xavier_initializer(), dtype=tf.float64 )

def biases(name, shape):
	return tf.get_variable(name, shape, initializer= tf.zeros_initializer(), dtype=tf.float64 )

def conv2d(x, W, strides=[1,2,2,1]):
	conv = tf.nn.conv2d(x, W, padding="SAME")
	return conv
	
def max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], name="Pool2"):
	with tf.variable_scope(name):
		m = tf.nn.max_pool2d(x, ksize=ksize, strides=strides, padding="SAME")
	return m

def fc_layer(x, W):
	f = tf.matmul(x, W)
	return f

def flatten(x):
	fshape = x.get_shape()[1] * x.get_shape()[2] * x.get_shape()[3]
	shape = [-1, fshape]
	fl = tf.reshape(x, shape, name="flatten")
	return fl

def get_fullshape(x):
	return tf.cast(x.get_shape(), 'int32' )

def get_shape(x, index, dtype='int32'):
	return tf.cast(x.get_shape()[index], dtype)
	
#Tensorboard utils:
def variable_summaries(var, name=""):
  with tf.name_scope(name + '_summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def display_activation(var, name="var", reshape_height = 4, resize_scale = 5):
	var = tf.slice(var, begin= [0,0,0,0], size=[1,get_shape(var,1),get_shape(var,2),get_shape(var,3) ] )
	var = tf.reshape(var, shape=(1, get_shape(var,1)*reshape_height, -1, 1) )
	# ~ var = tf.image.resize_images(var, size=(get_shape(var,1)*resize_scale,get_shape(var,2)*resize_scale) )
	tf.summary.image(name, var)

def display_activation_sep(var, name="var", filters= 2):
	with tf.name_scope(name):			
		var = tf.slice(var, begin= [0,0,0,0], size=[1,get_shape(var,1),get_shape(var,2),get_shape(var,3) ] )
		for filt in range(filters):				
			tmp = tf.slice(var, begin=[0,0,0,filt], size=(1, get_shape(var,1), get_shape(var, 2), 1) )
			tf.summary.image(name+ "_" + str(filt), tmp )
	
def display_image(var, name="var"):
	var = tf.slice(var, begin= [0,0,0,0], size=[1,get_shape(var,1),get_shape(var,2),get_shape(var,3) ] )
	tf.summary.image(name, var)


#Naming utilities
def addNameToTensor(X, name):
	return tf.identity(X, name=name)


def batch_norm(X, train_flag, decay=0.999, epsilon = 1e-3):

	scale = tf.Variable( tf.ones( [X.get_shape()[-1]] ) )
	beta = tf.Variable(tf.zeros( [X.get_shape()[-1]] ) )
	
	population_mean = tf.Variable( tf.zeros([X.get_shape()[-1]]), trainable=False )
	population_var = tf.Variable( tf.ones([X.get_shape()[-1]]), trainable=False )
	
	if train_flag == True:
		#Moments
		batch_mean, batch_var = tf.nn.moments( X, axes=[0, 1, 2])
		
		train_mean = tf.assign( population_mean, population_mean*decay +  batch_mean*(1-decay) )
		train_var = tf.assign( population_var, population_var*decay + batch_var*(1-decay) )
		
		with tf.control_dependencies( [train_mean, train_var] ):
			X_bn = tf.nn.batch_normalization( X, batch_mean, batch_var, beta, scale, epsilon)
	
	else:
		X_bn = tf.nn.batch_normalization( X, population_mean, population_var, beta, scale, epsilon )
	
	return X_bn


#Encode- BLUE block as per SegNet paper
def Conv_BN_Act_block(X, kernel=[3,3,1,4], strides=[1,2,2,1], name="Encoder_1", train_flag=True):	
	with tf.variable_scope(name):
		with tf.variable_scope("Conv"):				
			W1_conv = weight("W_conv",kernel)
			b1_conv = biases("b_conv", [1, kernel[-1] ])
			Z1_conv = conv2d(X, W1_conv, strides) + b1_conv
		
		with tf.variable_scope("BatchNorm"):
			Z1_bn = batch_norm(Z1_conv, train_flag, decay=0.999, epsilon = 1e-3)
			
		with tf.variable_scope("ReLU"):
			A1 = tf.nn.relu(Z1_bn)
	return A1

#Decode- Upsampling using Conv2D_transpose- (Deconvolve, but not exactly)
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
		#output_shape = [s[0], s[1]*strides[1], s[2]*strides[2], kernel[2] ]
		output_shape = tf.stack( [ tf.shape(X)[0] , s[1]*strides[1], s[2]*strides[2], kernel[2] ] )
		
		#output_shape = [-1, s[1]*strides[1], s[2]*strides[2], kernel[2] ]
		#output_shape = tf.shape(X)	#works
		W_convt = weight("W_convt", kernel)
		out = tf.nn.conv2d_transpose(X, W_convt, output_shape, strides, padding="SAME")
	return out




########################################################################
##	Build a CNN architecture
########################################################################
def IoU(logits, labels):
	#logits - [m, num1, num2, 1] & represents prob of class 1
	with tf.variable_scope("IoU"):			
		logits = tf.reshape(logits, [-1])	#flattens t
		labels = tf.reshape(labels, [-1])
		
		inter = tf.reduce_sum( tf.multiply(logits, labels) )
		union = tf.subtract(tf.add(logits, labels), tf.multiply(logits, labels) )	#A+B-AB
		union = tf.reduce_sum( union ) 
		
		iou = tf.div(inter, union)
	# but when using it as loss USE: 1 - IoU
	return iou


def compute_accuracy(Zout, Y):
	logits = tf.cast(Zout, tf.float64)
	labels = tf.cast(Y, tf.float64)
	
	with tf.variable_scope("Accuracy"):
		# ~ correct = tf.equal(tf.argmax(logits, 1) , tf.argmax(labels, 1) )
		# ~ accuracy = tf.reduce_mean(tf.cast(correct, tf.float32) )
		correct = tf.equal(logits , labels)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float64))
	return accuracy

#model graph
class Model:
	def __init__(self, num1 = 28, num2 = 28, nclass=3):
		
		# ~ strategy = tf.distribute.MirroredStrategy()
		# ~ with strategy.scope():
			# ~ with tf.variable_scope("Model"):
				# ~ self.define_model(num1, num2, nclass)
		with tf.variable_scope("Model"):
			self.define_model(num1, num2, nclass)					


	def define_model(self,num1, num2, nclass):

		with tf.device("/gpu:0"):
				
			#create placeholders
			self.X = tf.placeholder(tf.float64, [None, num1, num2, 1], name="X")
			self.X = self.X/255.0

			print("Model shape- ",self.X.shape)		
			display_image(self.X, name="X")
			
			self.Y = tf.placeholder(tf.float64, [None, num1, num2, 1], name="Y")
			display_image(self.Y, name="Y")
			
			self.train_flag = tf.placeholder(tf.bool, name="train_flag")
			
			#-----------------------------------------------------------------------
			#	Conv 1 layer + BN + ReLU layer
			#-----------------------------------------------------------------------
		with tf.device("/gpu:1"):
			
			
			A1 = Conv_BN_Act_block(self.X, kernel=[3,3,1,16], strides=[1,2,2,1], name="ConvBA_1", train_flag= self.train_flag)
			display_activation_sep(A1, name="A1", filters= 16)
			#tf.summary.histogram('ConvBA_1', A1)
			
			A2 = max_pool2d(A1, ksize=[1,2,2,1], strides=[1,2,2,1], name="Pool_2")
		
		with tf.device("/gpu:2"):
				
			#Encode2
			A3 = Conv_BN_Act_block(A2, kernel=[3,3,16, 8], strides=[1,2,2,1], name="ConvBA_3", train_flag= self.train_flag)		
			A4 = max_pool2d(A3, ksize=[1,2,2,1], strides=[1,2,2,1], name="Pool_4")
				
			# ~ display_activation(A2_conv, name="A2_conv", reshape_height = 4, resize_scale = 5)
			#display_activation_sep(A2_conv, name="A2_conv", filters= 2)		
		
		with tf.device("/gpu:3"):
				
			#Encode3
			A5 = Conv_BN_Act_block(A4, kernel=[3,3,8,8], strides=[1,2,2,1], name="ConvBA_5", train_flag= self.train_flag)
			A6 = max_pool2d(A5, ksize=[1,2,2,1], strides=[1,2,2,1], name="Pool_6")
		
		with tf.device("/gpu:4"):
			
			#Decode4
			A7 = conv2d_transpose(A6, kernel=[3,3,8,8], strides=[1,2,2,1], name="ConvTrans_7")
			A8 = Conv_BN_Act_block(A7, kernel=[3,3,8,4], strides=[1,2,2,1], name="ConvBA_8", train_flag= self.train_flag)
		
		with tf.device("/gpu:5"):
							
			#Decode5
			A9 = conv2d_transpose(A8, kernel=[3,3,4,4], strides=[1,2,2,1], name="ConvTrans_9")
			A10 = Conv_BN_Act_block(A9, kernel=[3,3,4,4], strides=[1,2,2,1], name="ConvBA_10", train_flag= self.train_flag)
		
			#Decode6
			A11 = conv2d_transpose(A10, kernel=[3,3,nclass,4], strides=[1,2,2,1], name="ConvTrans_11")
			tf.summary.histogram('A11', A11)
			display_image(A11, name="A11")
			
			########
			self.logits = tf.identity(A11, name = "logits")
			print("Logits : ", self.logits.shape)
			print("Logits : ", self.logits.dtype)		
			tf.summary.histogram('logits', self.logits)
			display_image(self.logits, name="logits")
			
		with tf.device("/gpu:6"):
			
			# ~ self.logits = tf.nn.sigmoid( A11, name="logits")
			self.predict = tf.argmax( tf.nn.softmax(A11), axis= 3)
			self.predict = tf.cast(self.predict, dtype=tf.float64)
			self.predict = tf.expand_dims(self.predict, axis=3, name="predict")
			tf.summary.histogram("Predict", self.predict)
			display_image(self.predict, name="predict")
						
			#============#
			print("A1 : ", A1.shape)
			print("A2 : ", A2.shape)
			print("A3 : ", A3.shape)
			print("A4 : ", A4.shape)
			print("A5 : ", A5.shape)
			print("A6 : ", A6.shape)
			print("A7 : ", A7.shape)
			print("A8 : ", A8.shape)
			print("A9 : ", A9.shape)
			print("A10 : ", A10.shape)
			print("A11 : ", A11.shape)
			print("Logits : ", self.logits.shape)
			#============#
			print("Logits: ", self.logits.shape)
			print("Labels: ", self.Y.shape)
		
			self.Y_hot = tf.stop_gradient( tf.one_hot( tf.cast(self.Y , tf.uint8), axis=-1, depth= nclass) )
			
			self.loss_raw = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_hot, logits=self.logits)
			self.loss = tf.reduce_sum(self.loss_raw, name="loss")
			tf.summary.scalar("Cross Entropy loss", self.loss)
			
			self.accuracy = compute_accuracy( tf.one_hot( tf.cast(self.predict, tf.uint8 ), axis=-1, depth=nclass) , self.Y_hot)
			#self.accuracy = tf.identity(self.loss, name="accuracy")
			tf.summary.scalar("accuracy", self.accuracy)
		
			#Project predictions
			self.logits_c1 = tf.cast( tf.math.equal(self.predict, tf.constant(1.0, dtype=tf.float64) ), dtype=tf.float64, name="logits_c1")
			display_image(self.logits_c1, name="logits_c1")
			self.logits_c2 = tf.cast( tf.math.equal(self.predict, tf.constant(2.0, dtype=tf.float64) ), dtype=tf.float64, name="logits_c2")
			display_image(self.logits_c2, name="logits_c2")
			self.logits_c3 = tf.cast( tf.math.equal(self.predict, tf.constant(3.0, dtype=tf.float64) ), dtype=tf.float64, name="logits_c3")
			display_image(self.logits_c3, name="logits_c3")
			
			#Project labels
			self.Y_c1 = tf.cast( tf.math.equal(self.Y, tf.constant(1.0, dtype=tf.float64) ), dtype=tf.float64, name="Y_c1")
			display_image(self.Y_c1, name="Y_c1")
			self.Y_c2 = tf.cast( tf.math.equal(self.Y, tf.constant(2.0, dtype=tf.float64) ), dtype=tf.float64, name="Y_c2")
			display_image(self.Y_c2, name="Y_c2")
			self.Y_c3 = tf.cast( tf.math.equal(self.Y, tf.constant(3.0, dtype=tf.float64) ), dtype=tf.float64, name="Y_c3")
			display_image(self.Y_c3, name="Y_c3")
	
		with tf.device("/gpu:7"):
	
			self.optimizer = tf.train.RMSPropOptimizer(learning_rate= 1e-2).minimize(self.loss_raw)
			self.var_init = tf.global_variables_initializer()
			
			self.merged = tf.summary.merge_all()
		
		
		##################################################################
		# ~ iou = IoU(self.Y, self.logits)
		# ~ self.loss = tf.subtract( tf.constant(1.0, dtype=tf.float64), iou, name="loss")
		# ~ tf.summary.scalar("IoU loss", self.loss)
		
		# ~ self.accuracy = compute_accuracy(self.logits, self.Y)
		# ~ tf.summary.scalar("accuracy", self.accuracy)		
		##################################################################
	
		#Loss
		# ~ iou_c1 = IoU(self.Y_c1, self.logits_c1)
		# ~ iou_c2 = IoU(self.Y_c2, self.logits_c2)
		# ~ iou_c3 = IoU(self.Y_c3, self.logits_c3)
		# ~ iou = iou_c1 + iou_c2 + iou_c3
		
		# ~ tf.summary.scalar("IoU C1", iou_c1)
		# ~ tf.summary.scalar("IoU C2", iou_c2)
		# ~ tf.summary.scalar("IoU C3", iou_c3)


		# ~ self.loss = tf.subtract( tf.constant(3.0, dtype=tf.float64), iou, name="loss")
		# ~ tf.summary.scalar("IoU loss", self.loss)			
		
		#Accuracy
		# ~ self.accuracy_c1 = compute_accuracy(self.logits_c1, self.Y_c1)
		# ~ tf.summary.scalar("accuracy_c1", self.accuracy_c1)
		# ~ self.accuracy_c2 = compute_accuracy(self.logits_c2, self.Y_c2)
		# ~ tf.summary.scalar("accuracy_c2", self.accuracy_c2)
		# ~ self.accuracy_c3 = compute_accuracy(self.logits_c3, self.Y_c3)
		# ~ tf.summary.scalar("accuracy_c3", self.accuracy_c3)
		
		# ~ self.accuracy = (self.accuracy_c1 + self.accuracy_c2 + self.accuracy_c3)/3.0
		# ~ tf.summary.scalar("accuracy", self.accuracy)
		
		# ~ self.optimizer = tf.train.RMSPropOptimizer(learning_rate= 1e-1).minimize(self.loss)
		#with tf.device("/gpu:2"):
		
			
	def test(self, sess, x_batch, y_batch):
		 test_loss, test_accu, test_summary, logits = sess.run([self.loss, self.accuracy, self.merged, self.logits], feed_dict={self.X: x_batch, self.Y: y_batch, self.train_flag: False} )
		 return test_loss, test_accu, test_summary

	def train(self, sess, x_batch, y_batch):
		summary, loss, _, accu, logits = sess.run( [self.merged, self.loss, self.optimizer, self.accuracy, self.logits], feed_dict={self.X: x_batch, self.Y: y_batch, self.train_flag: True} )
		return summary, loss, accu
	
def read_js(fname="./data_pairs.json"):
	with open(fname, 'r') as infile:
		data = json.load(infile)
	return data

def write_js(data, fname="./data_pairs.json"):
	with open(fname, 'w') as outfile:
		json.dump(fp=outfile, obj=data, indent=4)

def allocate():
	# ~ config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 0})
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	config.gpu_options.allow_growth=True
	return config

def read_image(fname):
	return cv2.imread(fname, 0)

class Manager:
	def __init__(self, sess, directory= "./dataset/", fmt="png", outfold="./out1/", batch_size= 64, shuffle=True, fetch_size= 5000, num1=512, num2=512, nclass=3):
		self.directory = directory
		
		self.Xdirectory = self.directory + "/X/"
		self.Ydirectory = self.directory + "/Y/"
		
		#X only
		self.fnames = os.listdir(self.Xdirectory)
		
		#Filter images that aint the fmt
		j = 0
		while j < len(self.fnames):
			fname = self.fnames[j]
			if fname.endswith("."+fmt):
				j = j+1
			else:
				self.fnames.pop(j)
		
		self.outfold = outfold
		
		#params
		self.num1 = num1
		self.num2 = num2
		self.nclass = nclass
		self.batch_size = batch_size
		self.ratio = [0.8, 0.1, 0.1]
		
		#others
		self.len = len(self.fnames)
		self.inds = np.arange(self.len)
		print("inds -", self.inds)
		orig_len = deepcopy(self.len)

		if shuffle ==True:
			np.random.seed(4)
			np.random.shuffle(self.inds)
			tmp = []
			fcount = 0
			for i in self.inds:
				tmp.append( self.fnames[i] )
				#to filter excess dataset
				fcount = fcount+1
				if fcount == fetch_size:
					self.len = fetch_size
					break
					
			self.fnames = deepcopy(tmp)
		
		#split_names
		self.train_names, self.test_names, self.dev_names = self.split_names(self.fnames, self.len, self.ratio)

		#save split_names to outfolder
		jname = self.outfold + "/data_splits.json"
		jdata = {
				"Orig_len": orig_len,
				"fetch_size": fetch_size,
				"Xdirectory": self.Xdirectory,
				"Ydirectory": self.Ydirectory,
				"fnames_len": self.len,
				"train_names": self.train_names,
				"test_names": self.test_names,
				"dev_names": self.dev_names,
				}
		write_js(jdata, jname)
		
		m = len(self.train_names)
		self.total_minibatches = math.ceil(m/self.batch_size)
		mdev = len(self.dev_names)
		self.total_minibatches_dev = math.ceil(mdev/self.batch_size)
		mtest = len(self.test_names)
		self.total_minibatches_test = math.ceil(mtest/self.batch_size)
			
		print("Total train: ", self.total_minibatches)
		print("Total dev: ", self.total_minibatches_dev)
		print("Total test: ", self.total_minibatches_test)
		
		self.sess = sess
		
		self.mod = Model(num1 = self.num1 , num2 = self.num2, nclass= self.nclass)
		
		self.saver = tf.train.Saver(max_to_keep=100)

		#summary
		self.train_writer = tf.summary.FileWriter(self.outfold + "/logs/train/", self.sess.graph)
		self.test_writer = tf.summary.FileWriter(self.outfold + "/logs/test/")
		
		self.sess.run(self.mod.var_init)
		
	#Use within init
	def split_names(self, fnames, length, ratio=[0.8,0.1,0.1]):
		len_train = int(0.8*length)
		len_test = int(0.1*length)
		len_dev = int(0.1*length)
		
		train_names = fnames[:len_train]
		test_names = fnames[len_train: len_train+len_test]
		dev_names = fnames[len_train+len_test: ]
		
		return train_names, test_names, dev_names
			
	#fetch batch
	def get_x_batch_names(self, train_names, batch_num):
		if batch_num <= self.total_minibatches:
			x_batch_names = train_names[batch_num*self.batch_size: (batch_num+1)*self.batch_size ]
		else:
			print("Batch num overflow")
			exit(121)				
		return x_batch_names
	
	#read images
	def read_xy_batch_names(self, x_batch_names):
		outX = []
		outY = []
		for name in x_batch_names:
			outX.append(cv2.imread(self.Xdirectory+ "/" + name,0).reshape(self.num1, self.num2, 1) )
			outY.append(cv2.imread(self.Ydirectory+ "/" + name,0).reshape(self.num1, self.num2, 1) )
		return np.array(outX), np.array(outY)
	
	#Use during training: uses the above two functions- get & read x_batch_names
	def get_batch(self, train_names, batch_num):
		#1
		x_batch_names = self.get_x_batch_names(train_names, batch_num)
		#2
		x_batch, y_batch = self.read_xy_batch_names(x_batch_names)
		return x_batch, y_batch
	
	#TRAIN	
	def start_train(self, epochs=20):

		#Training
		plt.ion()		
		
		for epoch in tqdm(range(epochs)):
			epoch_cost = 0.0
			epoch_accu = 0.0	
			for batch_num in tqdm(range(self.total_minibatches)):
				x_train_batch, y_train_batch = self.get_batch(self.train_names, batch_num)
								
				train_summary, train_loss, train_accuracy = self.mod.train(self.sess, x_train_batch, y_train_batch)
				
				epoch_cost = epoch_cost + train_loss/self.total_minibatches
				epoch_accu = epoch_accu + train_accuracy/self.total_minibatches
				
				if batch_num%5 == 0:						
					self.train_writer.add_summary(train_summary, epoch)
			
			#Run accu for test
			#epoch_acc_test, test_summary = self.sess.run( [self.mod.accuracy, self.mod.merged], feed_dict={self.mod} )
			epoch_cost_dev = 0.0
			epoch_accu_dev = 0.0	
			for batch_num in tqdm(range(self.total_minibatches_dev)):
				x_dev_batch, y_dev_batch = self.get_batch(self.dev_names, batch_num)
				loss_dev, accu_dev, dev_summary = self.mod.test(self.sess, x_dev_batch, y_dev_batch)
				
				epoch_cost_dev = epoch_cost_dev + loss_dev/self.total_minibatches_dev
				epoch_accu_dev = epoch_accu_dev + accu_dev/self.total_minibatches_dev
				
				if batch_num%5 == 0:						
					self.test_writer.add_summary(dev_summary, epoch)

			plt.scatter(epoch, epoch_accu, c='r', label="train")
			plt.scatter(epoch, epoch_accu_dev, c='b', label="dev")
			plt.savefig(self.outfold + "/calculated_accu.png")
			plt.pause(0.01)
			
			# ~ if epoch%2 == 0:				
			#if epoch%2 == 0:
			self.saver.save(self.sess, self.outfold + '/model/segment', global_step=epoch)
			
			
			print("\n\nAt epoch {}, training cost: {} & Train Accuracy: {}".format(epoch, epoch_cost, epoch_accu) )
			print("\t testing cost: {} & Test Accuracy: {}".format(epoch_cost_dev, epoch_accu_dev) )
			
		plt.legend()
		plt.show()
		
		# ~ x_test_batch = self.read_x_batch_names(self.test_names)
		# ~ epoch_cost_test, epoch_accu_test, _ = self.mod.test(self.sess, x_test_batch)
		# ~ print("Test Performance: Cost= {} & Accu= {} ".format(epoch_cost_test, epoch_accu_test) )
		
		
		epoch_cost_test = 0.0
		epoch_accu_test = 0.0	
		for batch_num in tqdm(range(self.total_minibatches_test)):
			x_test_batch, y_test_batch = self.get_batch(self.test_names, batch_num)
			loss_test, accu_test, test_summary = self.mod.test(self.sess, x_test_batch, y_test_batch)
			
			epoch_cost_test = epoch_cost_test + loss_test/self.total_minibatches_test
			epoch_accu_test = epoch_accu_test + accu_test/self.total_minibatches_test
		print("Test Performance: Cost= {} & Accu= {} ".format(epoch_cost_test, epoch_accu_test) )
		
		
		#input("Save?")
		#tf.saved_model.simple_save(self.sess, self.outfold + "/model/", inputs={"X":self.mod.X, "train_flag":self.mod.train_flag}, outputs={"predict":self.mod.logits, "accuracy": self.mod.accuracy})
		
		
		perf_log = {
					"epoch_cost": float(np.sum(epoch_cost)),
					"epoch_accu": float(epoch_accu),
					"epoch_cost_dev": float(np.sum(epoch_cost_dev) ),
					"epoch_accu_dev": float(epoch_accu_dev),
					"epoch_cost_test": float( np.sum(epoch_cost_test) ),
					"epoch_accu_test": float(epoch_accu_test)
					}
		
		write_js(perf_log, fname= self.outfold + "/model_performance_logs.json")
		
		
		#######################################################
		
if __name__ == "__main__":
	
	#directory = "/home/ai-nano/Documents/McMaster_box/test/test_resize_read/"
	# ~ directory = "/home/aswin-rpi/Documents/GITs/test_resize/"
	# ~ outfold = "/home/aswin-rpi/Documents/GITs/test_resize_OUT_3/"
	
	directory = "/data/McMaster/raw_ready_resize/"
	outfold = "/data/McMaster/raw_ready_resize_OUT/Expt11_Test_8xRTX2080Ti/"

	
	outfold = outfold + "/" + str(len(glob.glob(outfold+"/*") ) ) + "/"
	
	fmt = "png"
	batch_size = 32
	shuffle = True
	fetch_size = 350000
	num1 = 512
	num2 = 512
	
	epochs = 15
	
	
	print("\n#### OUTFOLD is ", outfold)
	pathlib.Path(directory).mkdir(exist_ok=True, parents=True)
	pathlib.Path(outfold).mkdir(exist_ok=True, parents=True)
	
	with tf.Session(config=allocate() ) as sess:		
		manager = Manager( sess, directory, fmt, outfold, batch_size, shuffle, fetch_size, num1, num2)
		
		manager.start_train(epochs)
		
	tf.reset_default_graph()
	
