import numpy as np
import tensorflow as tf
import pathlib
import math
import pickle


class Predict_Manager:
	def __init__(self, data_pkl, mod_dir= "./folder/", gram_size=24, feature_size=7):
		with open(data_pkl, 'rb') as infile:
			self.data = pickle.load(infile)
		
		#1) Init data
		self.data_X = np.array(self.data["X"], dtype=np.float64)
		self.data_y = np.array(self.data["y"], dtype=np.float64)
		self.csv_out = self.data["csv_out"]
		print("In Manager, Xy shapes- ", self.data_X.shape, self.data_y.shape)
		
		self.load_model(mod_dir)
	
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

	def load_model(self, model_dir, only_predict=False):
		tf.reset_default_graph()
					
		graph1 = tf.Graph()
		with graph1.as_default():		
			self.sess = tf.Session(graph=graph1)
		
			print("##=============================================================##")
			print("Restoring...")
			
			#saved_model API- use the standard tag
			tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_dir)
			
			self.xs = graph1.get_tensor_by_name("Model/X:0")
			#self.ys = graph1.get_tensor_by_name("Y:0")
			self.keep_prob = graph1.get_tensor_by_name("Model/keep_prob:0")
			self.th = graph1.get_tensor_by_name("Model/th:0")
			
			self.pred = graph1.get_tensor_by_name("Model/predict:0")
			
		
			if only_predict == False:
				self.accuracy = graph1.get_tensor_by_name("Model/Accuracy/Mean:0")
				self.ys = graph1.get_tensor_by_name("Model/Y:0")
				

			
	def predict(self, x_batch, y_batch = None, only_predict=False, param = 0.03):
		if only_predict == True:
			out = self.sess.run(self.pred, feed_dict={self.xs: x_batch, self.keep_prob: 1.0, self.th:param} )
			return out
		else:
			y_batch = np.array(y_batch).reshape(-1,1)
			
			out, accu = self.sess.run([self.pred, self.accuracy], feed_dict={self.xs: x_batch, self.ys: y_batch, self.keep_prob: 1.0, self.th:param} )
			return (out, accu)
	
if __name__ == "__main__":
	
	pkl_name = "/media/aswin/Data/GITs/Kyma/Data/b4_GAF_Xy.pkl"
	#mod_dir = "/media/aswin/Data/GITs/Kyma/Data/CNN/v2/model/"
	mod_dir = "/media/aswin/Data/GITs/Kyma/Data/CNN/v3_b4_GAF_Xy/model/"
	
	param = 0.15
	
	manage = Predict_Manager(pkl_name, mod_dir)
	
	out, accu = manage.predict(x_batch = manage.data_X[:50], y_batch = manage.data_y[:50], only_predict=False, param=param)
	
	print("##### BAD #####")
	print("Out: ", out.ravel())
	print("y: ", manage.data_y.ravel()[:50])
	print("Accuracy: ", accu )
	print("Lens ", len(out.ravel()), len(manage.data_y.ravel()[:50]) )
	

	out, accu = manage.predict(x_batch = manage.data_X[-22:], y_batch = manage.data_y[-22:], only_predict=False, param=param)	
	print("\n##### GOOD #####")
	print("Out: ", out.ravel())
	print("y: ", manage.data_y.ravel()[-22:])
	print("Accuracy: ", accu)
	print("Lens ", len(out.ravel()), len(manage.data_y.ravel()[-22:]) )

##############################################################################
	#Batch 5 prediction
	# ~ pkl_name = "/media/aswin/Data/GITs/Kyma/Data/b5_GAF_Xy.pkl"
	# ~ mod_dir = "/media/aswin/Data/GITs/Kyma/Data/CNN/v2/model/"
	# ~ param = 0.09
	
	# ~ manage = Predict_Manager(pkl_name, mod_dir)
	
	# ~ truth_y = np.ones(len(manage.data_X))
	# ~ out, accu = manage.predict(x_batch = manage.data_X, y_batch = truth_y, only_predict=False, param=param)
	
	# ~ print("####### B5 ############")
	# ~ print("Out: ", out.ravel())
	# ~ print("y: ", truth_y)
	# ~ print("Accuracy: ", accu )
	# ~ print("Lens ", len(out.ravel()), len(truth_y) )
	
	
