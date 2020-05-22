import numpy as np
import tensorflow as tf
import pathlib
import math
import json
import glob
import matplotlib.pyplot as plt
import pickle
import h5py
from sklearn.utils import shuffle as sk_shuffle
np.random.seed(4)

#parameter utils
def weight(name, shape):
	return tf.get_variable(name, shape, initializer= tf.contrib.layers.xavier_initializer(), dtype=tf.float64 )

def biases(name, shape):
	return tf.get_variable(name, shape, initializer= tf.zeros_initializer(), dtype=tf.float64 )

def conv2d(x, W):
	c = tf.nn.conv2d(x, W, strides=[1,2,2,1], padding="SAME")
	return c
	
def max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1]):
	m = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding="SAME")
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
	
	# ~ var = tf.transpose(var, [3,0,1,2])
	#paddings is off for now
	# ~ paddings = tf.constant([[0,0],[2,2], [2, 2],[0,0]])
	# ~ var = tf.pad(var, paddings, "CONSTANT")
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
	

########################################################################
##	Build a CNN architecture
########################################################################
def compute_accuracy(Zout, Y):
	logits = tf.cast(Zout, tf.float64)
	labels = Y
	with tf.name_scope("Accuracy"):
		# ~ correct = tf.equal(tf.argmax(logits, 1) , tf.argmax(labels, 1) )
		# ~ accuracy = tf.reduce_mean(tf.cast(correct, tf.float32) )
		correct = tf.equal(logits , labels)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	return accuracy

#model graph
class Model:
	def __init__(self, n_out, num1 = 28, num2 = 28, feature_size=7):
		
		with tf.variable_scope("Model"):
			self.define_model(num1, num2, feature_size, n_out)
			
	def define_model(self,num1, num2, feature_size, n_out):
		pooling = False
		#create placeholders
		self.X = tf.placeholder(tf.float64, [None, feature_size, num1, num2], name="X")
		print(self.X.shape)
		self.X1 = tf.reshape(self.X, shape=[-1, feature_size*num1, num2, 1], name="X_shaped")

		display_image(self.X1, name="X1")
		
		self.Y = tf.placeholder(tf.float64, [None, n_out], name="Y")
		self.keep_prob = tf.placeholder(tf.float64, name="keep_prob")

		self.th = tf.placeholder(tf.float64, name="th")

		#-----------------------------------------------------------------------
		#	Conv 1 layer + pool layer
		#-----------------------------------------------------------------------
		with tf.name_scope("Conv1"):
			W1_conv = weight("W1_conv",[3,3,1,4])
			b1_conv = biases("b1_conv", [1, 4])
			A1_conv = tf.nn.relu(conv2d(self.X1, W1_conv) + b1_conv)
		display_activation_sep(A1_conv, name="A1_conv", filters= 4)
		
		tf.summary.histogram('A1_conv', A1_conv)
		
		if pooling == True:				
			with tf.name_scope("Pool1"):
				A1_pool = max_pool2d(A1_conv, ksize=[1,2,2,1])
		else:
			A1_pool = A1_conv
			
		#-----------------------------------------------------------------------
		#	Conv 2 layer + pool layer
		#-----------------------------------------------------------------------
		with tf.name_scope("Conv2"):
			W2_conv = weight("W2_conv",[3,3,4, 2])
			b2_conv = biases("b2_conv", [1, 2])
			A2_conv = tf.nn.relu(conv2d(A1_pool, W2_conv) + b2_conv)
		# ~ display_activation(A2_conv, name="A2_conv", reshape_height = 4, resize_scale = 5)
		display_activation_sep(A2_conv, name="A2_conv", filters= 2)
		
		if pooling == True:				
			with tf.name_scope("Pool2"):
				A2_pool = max_pool2d(A2_conv, ksize=[1,2,2,1])
		else:
			A2_pool = A2_conv
		
		# ~ #-----------------------------------------------------------------------
		# ~ #	Conv 3 layer + pool layer
		# ~ #-----------------------------------------------------------------------
		# ~ with tf.name_scope("Conv3"):
			# ~ W3_conv = weight("W3_conv",[3,3,8,16])
			# ~ b3_conv = biases("b3_conv", [1, 16])
			# ~ A3_conv = tf.nn.relu(conv2d(A2_pool, W3_conv) + b3_conv)
		# ~ display_activation_sep(A3_conv, name="A3_conv", filters= 16)
		
		# ~ if pooling == True:				
			# ~ with tf.name_scope("Pool3"):
				# ~ A3_pool = max_pool2d(A3_conv, ksize=[1,2,2,1])
		# ~ else:
			# ~ A3_pool = A3_conv
		
		#-----------------------------------------------------------------------
		#	Flatten
		#-----------------------------------------------------------------------
		#A_flatten = flatten(A4_pool)
		self.fv = flatten(A2_pool)
		tf.summary.histogram('fv', self.fv)
		
		#-----------------------------------------------------------------------
		#	fc1
		#-----------------------------------------------------------------------
		with tf.name_scope("fc1"):
			#W_fc1 = weight("W_fc1", [get_shape(A_flatten, index=1), 1024])
			W_fc1 = weight("W_fc1", [504, 32])
			b_fc1 = biases("b_fc1", [1,32])
			A_fc1 = tf.nn.relu(fc_layer(self.fv, W_fc1) + b_fc1 )	

		tf.summary.histogram('A_fc1', A_fc1)
		#-----------------------------------------------------------------------
		#	Dropout
		#-----------------------------------------------------------------------
		with tf.name_scope("Dropout"):
			A_dropout = tf.nn.dropout(A_fc1, self.keep_prob)
		#-----------------------------------------------------------------------
		#	fc2
		#-----------------------------------------------------------------------
		with tf.name_scope("fc2"):
			#W_fc2 = weight("W_fc2", [get_shape(A_flatten, index=1), 2])
			W_fc2 = weight("W_fc2", [32, n_out])
			b_fc2 = biases("b_fc2", [1,n_out])
			Z_fc2 = tf.add( fc_layer(A_dropout, W_fc2) , b_fc2, name="Zout")
			#This is Z , not A
			#self.logits = tf.nn.softmax(Z_fc2, name="logits")
			self.logits = Z_fc2
		
		tf.summary.histogram('logits', self.logits)
		
		
		self.predict = tf.to_int32( self.logits > self.th, name = "predict")
		tf.summary.histogram('predict', self.predict)
		
		
		#self.Y_hot = tf.one_hot(tf.cast(tf.reshape(self.Y, [-1]) , tf.uint8) , depth= n_out)
		#print("Y_hot shape", self.Y_hot.shape)
		
		self.loss = tf.losses.mean_squared_error(self.Y , self.logits)
		#self.loss = tf.keras.losses.categorical_crossentropy(self.Y, self.logits)
		tf.summary.scalar("loss", self.loss)
		
		self.accuracy = compute_accuracy(self.predict, self.Y)
		tf.summary.scalar("accuracy", self.accuracy)

		self.optimizer = tf.train.RMSPropOptimizer(learning_rate= 1e-2).minimize(self.loss)
		self.var_init = tf.global_variables_initializer()
		
		self.merged = tf.summary.merge_all()
		
	def test(self, sess, x_batch, y_batch):
		 test_loss, test_accu, test_summary, logits = sess.run([self.loss, self.accuracy, self.merged, self.logits], feed_dict={self.X: x_batch, self.Y: y_batch, self.keep_prob: 1.0, self.th: 0.03} )
		 return test_loss, test_accu, test_summary
	
	def train(self, sess, x_batch, y_batch):
		summary, loss, _, accu, logits = sess.run( [self.merged, self.loss, self.optimizer, self.accuracy, self.logits], feed_dict={self.X: x_batch, self.Y: y_batch, self.keep_prob: 1.0, self.th: 0.03} )
		# ~ print("Train logits: ", logits)
		# ~ print("Labels ", y_batch)
		# ~ print("Loss ", loss)
		# ~ input("Waity")
		return summary, loss, accu
	
def read_js(fname="./data_pairs.json"):
	with open(fname, 'r') as infile:
		data = json.load(infile)
	return data

def write_js(data, fname="./data_pairs.json"):
	with open(fname, 'w') as outfile:
		json.dump(fp=outfile, obj=data, indent=2)

def allocate():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	return config

class Manager:
	def __init__(self, sess, data_pkl, directory= "./folder/", gram_size=24, feature_size=7):
		self.directory = directory
		
		with open(data_pkl, 'rb') as infile:
			self.data = pickle.load(infile)
		
		#1) Init data
		self.data_X = np.array(self.data["X"], dtype=np.float64)
		self.data_y = np.array(self.data["y"], dtype=np.float64)
		self.csv_out = self.data["csv_out"]
		print("In Manager, Xy shapes- ", self.data_X.shape, self.data_y.shape)
		
		#2) Shuffle and Split
		## len of ratio decides no_dev: [0.8,0.1,0.1] or [0.8,0.2]
		splits = self.split_dataset(self.data_X, self.data_y, ratio= [0.8,0.1, 0.1] , shuffle=True)
		self.train_X, self.train_y, self.dev_X, self.dev_y, self.test_X, self.test_y = splits
		
		#save data to outfolder
		jname = self.directory + "/data_splits.json"
		jdata = {
				"train_X": self.train_X.tolist(),
				"train_y": self.train_y.tolist(),
				"dev_X": self.dev_X.tolist(),
				"dev_y": self.dev_y.tolist(),
				"test_X": self.test_X.tolist(),
				"test_y": self.test_y.tolist()
				}
		write_js(jdata, jname)
		
		print("Train: ", self.train_X.shape, self.train_y.shape)
		print("Dev: ", self.dev_X.shape, self.dev_y.shape)
		print("Test: ", self.test_X.shape, self.test_y.shape)
		
		self.batch_size = 6
		n_out = self.train_y.shape[1]

		self.gram_size = gram_size
		self.feature_size= feature_size
		
		m = self.train_X.shape[0]
		self.total_minibatches = math.ceil(m/self.batch_size)
		print("Total: ", self.total_minibatches)
		
		self.sess = sess
		
		self.mod = Model(n_out, num1 = gram_size , num2 = gram_size, feature_size= feature_size)
		#summary
		self.train_writer = tf.summary.FileWriter(self.directory + "/logs/train/", self.sess.graph)
		self.test_writer = tf.summary.FileWriter(self.directory + "/logs/test/")
		
		self.sess.run(self.mod.var_init)
	
	def split_dataset(self, data_X, data_y, ratio= [0.8,0.1, 0.1] , shuffle=True):
		
		if shuffle == True:
			data_X, data_y = sk_shuffle(data_X, data_y, random_state=7)	#sklearn
			
		m = data_X.shape[0]
		
		if len(ratio) == 3:
			j = 1
			while j<len(ratio):
				ratio[j] = ratio[j] + ratio[j-1]
				j = j+1
			
			train_X = data_X[:int(m*ratio[0]) ]
			train_y = data_y[:int(m*ratio[0]) ]
			
			dev_X = data_X[int(m*ratio[0]): int(m*ratio[1])]
			dev_y = data_y[int(m*ratio[0]): int(m*ratio[1])]
			
			test_X = data_X[int(m*ratio[1]): int(m*ratio[2])]
			test_y = data_y[int(m*ratio[1]): int(m*ratio[2])]
			
			return train_X, train_y, dev_X, dev_y, test_X, test_y
		else:
			train_X = data_X[:int(m*ratio[0]) ]
			train_y = data_y[:int(m*ratio[0]) ]
			
			dev_X = data_X[int(m*ratio[0]): int(m*ratio[1])]
			dev_y = data_y[int(m*ratio[0]): int(m*ratio[1])]
			
			return train_X, train_y, dev_X, dev_y
	
	#fetch batch
	def get_x_batch(self, data_x, batch_num):
		if batch_num <= self.total_minibatches:
			x_batch = data_x[batch_num*self.batch_size: (batch_num+1)*self.batch_size ]
		else:
			print("Batch num overflow")
			exit(121)				
		return x_batch
		
	def get_xy_batch(self, X,y,batch_num):
		batch_x = self.get_x_batch(X, batch_num)
		batch_y = self.get_x_batch(y, batch_num)
		return batch_x, batch_y
		
	def start_train(self, epochs=20):

		#Training
		plt.ion()
		
		for epoch in range(epochs):
			epoch_cost = 0.0
			epoch_accu = 0.0	
			for batch_num in range(self.total_minibatches):
				x_batch, y_batch = self.get_xy_batch(self.train_X, self.train_y, batch_num)
				train_summary, train_loss, train_accuracy = self.mod.train(self.sess, x_batch, y_batch)
				# ~ print("Man Train loss", train_loss)
				# ~ input("wait")
				
				epoch_cost = epoch_cost + train_loss/self.total_minibatches
				epoch_accu = epoch_accu + train_accuracy/self.total_minibatches
			
			#Run accu for test
			#epoch_acc_test, test_summary = self.sess.run( [self.mod.accuracy, self.mod.merged], feed_dict={self.mod} )
			epoch_cost_dev, epoch_accu_dev, dev_summary = self.mod.test(self.sess, self.dev_X, self.dev_y)
			
			plt.scatter(epoch, epoch_accu, c='r', label="train")
			plt.scatter(epoch, epoch_accu_dev, c='b', label="dev")
			plt.pause(0.01)
			
			# ~ if epoch%2 == 0:				
			self.train_writer.add_summary(train_summary, epoch)
			self.test_writer.add_summary(dev_summary, epoch)	
			
			print("At epoch {}, training cost: {} & Train Accuracy: {}".format(epoch, epoch_cost, epoch_accu) )
			print("\t testing cost: {} & Test Accuracy: {}".format(epoch_cost_dev, epoch_accu_dev) )
		
		plt.legend()
		plt.show()
				
		epoch_cost_test, epoch_accu_test, _ = self.mod.test(self.sess, self.test_X, self.test_y)
		print("Test Performance: Cost= {} & Accu= {} ".format(epoch_cost_test, epoch_accu_test) )
		
		input("Save?")

		tf.saved_model.simple_save(sess, self.directory + "/model/", inputs={"X":self.mod.X, "Y": self.mod.Y, "keep_prob":self.mod.keep_prob, "th":self.mod.th}, outputs={"predict":self.mod.predict, "accuracy": self.mod.accuracy})
		
		
		perf_log = {
					"epoch_cost": float(epoch_cost),
					"epoch_accu": float(epoch_accu),
					"epoch_cost_dev": float(epoch_cost_dev),
					"epoch_accu_dev": float(epoch_accu_dev),
					"epoch_cost_test": float(epoch_cost_test),
					"epoch_accu_test": float(epoch_accu_test)
					}
		
		write_js(perf_log, fname= self.directory + "/model_performance_logs.json")
				
		
		
		#######################################################
		
if __name__ == "__main__":
	
	data_pkl = "/media/aswin/Data/GITs/Kyma/Data/b4_GAF_Xy.pkl"
	directory = "/media/aswin/Data/GITs/Kyma/Data/CNN/v4_b4_GAF_Xy/"
	epochs = 100
	
	pathlib.Path(directory).mkdir(exist_ok=True, parents=True)
	
	with tf.Session(config=allocate() ) as sess:		
		manager = Manager( sess, data_pkl, directory, gram_size=24, feature_size=7)
		
		manager.start_train(epochs)
		

	
