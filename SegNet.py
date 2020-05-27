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

def get_shape(x, index):
	return tf.cast(x.get_shape()[index], 'int32')
	
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
	#logits = tf.cast(Zout, tf.float64)
	logits = Zout
	labels = Y
	with tf.variable_scope("Accuracy"):
		# ~ correct = tf.equal(tf.argmax(logits, 1) , tf.argmax(labels, 1) )
		# ~ accuracy = tf.reduce_mean(tf.cast(correct, tf.float32) )
		correct = tf.equal(logits , labels)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	return accuracy

#model graph
class Model:
	def __init__(self, num1 = 28, num2 = 28):
		
		with tf.variable_scope("Model"):
			self.define_model(num1, num2)
					
	def define_model(self,num1, num2):
		
		#create placeholders
		self.X = tf.placeholder(tf.float64, [None, num1, num2, 1], name="X")
		self.X = self.X/255.0

		print("Model shape- ",self.X.shape)		
		display_image(self.X, name="X")
		
		self.Y = tf.stop_gradient( tf.cast( tf.math.greater(self.X, tf.constant(0.0, dtype=tf.float64) ), dtype=tf.float64), name="Y" )
		display_image(self.Y, name="Y")
		
		self.train_flag = tf.placeholder(tf.bool, name="train_flag")
		
		#-----------------------------------------------------------------------
		#	Conv 1 layer + BN + ReLU layer
		#-----------------------------------------------------------------------
		
		A1 = Conv_BN_Act_block(self.X, kernel=[3,3,1,128], strides=[1,2,2,1], name="ConvBA_1", train_flag= self.train_flag)
		display_activation_sep(A1, name="A1", filters= 128)
		#tf.summary.histogram('ConvBA_1', A1)
		
		A2 = max_pool2d(A1, ksize=[1,2,2,1], strides=[1,2,2,1], name="Pool_2")
		
		#Encode2
		A3 = Conv_BN_Act_block(A2, kernel=[3,3,128,64], strides=[1,2,2,1], name="ConvBA_3", train_flag= self.train_flag)		
		A4 = max_pool2d(A3, ksize=[1,2,2,1], strides=[1,2,2,1], name="Pool_4")
		
		# ~ display_activation(A2_conv, name="A2_conv", reshape_height = 4, resize_scale = 5)
		#display_activation_sep(A2_conv, name="A2_conv", filters= 2)		

		#Encode3
		A5 = Conv_BN_Act_block(A4, kernel=[3,3,64,32], strides=[1,2,2,1], name="ConvBA_5", train_flag= self.train_flag)
		A6 = max_pool2d(A5, ksize=[1,2,2,1], strides=[1,2,2,1], name="Pool_6")
		
		#Decode4
		A7 = conv2d_transpose(A6, kernel=[3,3,16,32], strides=[1,2,2,1], name="ConvTrans_7")
		A8 = Conv_BN_Act_block(A7, kernel=[3,3,16,16], strides=[1,2,2,1], name="ConvBA_8", train_flag= self.train_flag)
		
		#Decode5
		A9 = conv2d_transpose(A8, kernel=[3,3,8,16], strides=[1,2,2,1], name="ConvTrans_9")
		A10 = Conv_BN_Act_block(A9, kernel=[3,3,8,4], strides=[1,2,2,1], name="ConvBA_10", train_flag= self.train_flag)
		
		#Decode6
		A11 = conv2d_transpose(A10, kernel=[3,3,1,4], strides=[1,2,2,1], name="ConvTrans_11")
		tf.summary.histogram('A11', A11)
		display_image(A11, name="A11")
		
		########
		self.logits = tf.nn.sigmoid( A11, name="logits")
		tf.summary.histogram('logits', self.logits)
		display_image(self.logits, name="logits")
		
		self.logits_th = tf.cast( tf.math.greater(self.logits, tf.constant(0.5, dtype=tf.float64) ), dtype=tf.float64, name="logits_th")
		tf.summary.histogram('logits_th', self.logits_th)
		display_image(self.logits_th, name="logits_th")
		
		
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
		
		#============#
		
		# ~ self.predict = tf.to_int32( self.logits > self.th, name = "predict")
		# ~ tf.summary.histogram('predict', self.predict)
		
		
		#self.Y_hot = tf.one_hot(tf.cast(tf.reshape(self.Y, [-1]) , tf.uint8) , depth= n_out)
		#print("Y_hot shape", self.Y_hot.shape)
		
		print("Logits: ", self.logits.shape)
		print("Labels: ", self.Y.shape)
		
		#self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y , logits= self.logits)
		#self.loss = tf.reduce_sum(self.loss)
		#tf.summary.scalar("loss", tf.reduce_sum(self.loss) )
		
		iou = IoU(self.Y, self.logits)
		self.loss = tf.subtract( tf.constant(1.0, dtype=tf.float64), iou, name="loss")
		tf.summary.scalar("IoU loss", self.loss)
		
		self.accuracy = compute_accuracy(self.logits, self.Y)
		tf.summary.scalar("accuracy", self.accuracy)

		self.optimizer = tf.train.RMSPropOptimizer(learning_rate= 1e-2).minimize(self.loss)
		self.var_init = tf.global_variables_initializer()
		
		self.merged = tf.summary.merge_all()
		
	def test(self, sess, x_batch):
		 test_loss, test_accu, test_summary, logits = sess.run([self.loss, self.accuracy, self.merged, self.logits], feed_dict={self.X: x_batch, self.train_flag: False} )
		 return test_loss, test_accu, test_summary

	def train(self, sess, x_batch):
		summary, loss, _, accu, logits = sess.run( [self.merged, self.loss, self.optimizer, self.accuracy, self.logits], feed_dict={self.X: x_batch, self.train_flag: True} )
		return summary, loss, accu
	
def read_js(fname="./data_pairs.json"):
	with open(fname, 'r') as infile:
		data = json.load(infile)
	return data

def write_js(data, fname="./data_pairs.json"):
	with open(fname, 'w') as outfile:
		json.dump(fp=outfile, obj=data, indent=4)

def allocate():
	config = tf.ConfigProto(log_device_placement=True)
	config.gpu_options.allow_growth=True
	return config

def read_image(fname):
	return cv2.imread(fname, 0)

class Manager:
	def __init__(self, sess, directory= "./dataset/", fmt="png", outfold="./out1/", batch_size= 64, shuffle=True, fetch_size= 5000, num1=512,num2=512):
		self.directory = directory
		self.fnames = glob.glob(self.directory + "/*."+ fmt)
		self.outfold = outfold
		
		#params
		self.num1 = num1
		self.num2 = num2
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
				"fnames_len": self.len,
				"train_names": self.train_names,
				"test_names": self.test_names,
				"dev_names": self.dev_names,
				}
		write_js(jdata, jname)
		
		
		m = len(self.train_names)
		self.total_minibatches = math.ceil(m/self.batch_size)
		print("Total: ", self.total_minibatches)
		
		self.sess = sess
		
		self.mod = Model(num1 = self.num1 , num2 = self.num2)
		
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
	def read_x_batch_names(self, x_batch_names):
		out = []
		for name in x_batch_names:
			out.append(cv2.imread(name,0).reshape(self.num1, self.num2, 1) )
		return np.array(out)
	
	#Use during training: uses the above two functions- get & read x_batch_names
	def get_batch(self, train_names, batch_num):
		#1
		x_batch_names = self.get_x_batch_names(train_names, batch_num)
		#2
		x_batch = self.read_x_batch_names(x_batch_names)
		return x_batch
	
	
	#TRAIN	
	def start_train(self, epochs=20):

		#Training
		# ~ plt.ion()		
		
		for epoch in tqdm(range(epochs)):
			epoch_cost = 0.0
			epoch_accu = 0.0	
			for batch_num in tqdm(range(self.total_minibatches)):
				x_train_batch = self.get_batch(self.train_names, batch_num)
				
				# ~ tmpX = self.sess.run(self.mod.X, feed_dict={self.mod.X: x_train_batch} )
				# ~ tmpY = self.sess.run(self.mod.Y, feed_dict={self.mod.X: x_train_batch} )
				
				# ~ print("\n## X: ", tmpX.min(), tmpX.max())
				# ~ print("\n## Y: ", tmpY.min(), tmpY.max() )
				# ~ input("Waity")
				
				train_summary, train_loss, train_accuracy = self.mod.train(self.sess, x_train_batch)
				
				epoch_cost = epoch_cost + train_loss/self.total_minibatches
				epoch_accu = epoch_accu + train_accuracy/self.total_minibatches
			
			#Run accu for test
			#epoch_acc_test, test_summary = self.sess.run( [self.mod.accuracy, self.mod.merged], feed_dict={self.mod} )
			x_dev_batch = self.read_x_batch_names(self.dev_names)
			epoch_cost_dev, epoch_accu_dev, dev_summary = self.mod.test(self.sess, x_dev_batch)
			
			# ~ plt.scatter(epoch, epoch_accu, c='r', label="train")
			# ~ plt.scatter(epoch, epoch_accu_dev, c='b', label="dev")
			# ~ plt.pause(0.01)
			
			# ~ if epoch%2 == 0:				
			#if epoch%2 == 0:
			self.saver.save(self.sess, self.outfold + '/model/segment', global_step=epoch)
			
			self.train_writer.add_summary(train_summary, epoch)
			self.test_writer.add_summary(dev_summary, epoch)	
			
			print("\n\nAt epoch {}, training cost: {} & Train Accuracy: {}".format(epoch, epoch_cost, epoch_accu) )
			print("\t testing cost: {} & Test Accuracy: {}".format(epoch_cost_dev, epoch_accu_dev) )
			
		# ~ plt.legend()
		# ~ plt.show()
		
		x_test_batch = self.read_x_batch_names(self.test_names)
		epoch_cost_test, epoch_accu_test, _ = self.mod.test(self.sess, x_test_batch)
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
	directory = "/home/aswin-rpi/Documents/GITs/test_resize/"
	outfold = "/home/aswin-rpi/Documents/GITs/test_resize_OUT_3/"
	
	# ~ directory = "/home/aswin-rpi/Documents/GITs/McMaster/raw_X_resize/"
	# ~ outfold = "/home/aswin-rpi/Documents/GITs/McMaster/raw_X_resize_Outfold/"
	
	
	outfold = outfold + "/" + str(len(glob.glob(outfold+"/*") ) ) + "/"
	
	fmt = "png"
	batch_size = 4
	shuffle = True
	fetch_size = 14
	num1 = 512
	num2 = 512
	
	epochs = 2
	
	
	pathlib.Path(directory).mkdir(exist_ok=True, parents=True)
	pathlib.Path(outfold).mkdir(exist_ok=True, parents=True)
	
	with tf.Session(config=allocate() ) as sess:		
		manager = Manager( sess, directory, fmt, outfold, batch_size, shuffle, fetch_size, num1, num2)
		
		manager.start_train(epochs)
		

	
