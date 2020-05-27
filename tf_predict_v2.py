import numpy as np
import tensorflow as tf
import pathlib
import os
import cv2
import matplotlib.pyplot as plt


def allocate():
	config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 0})
	config.gpu_options.allow_growth=True
	return config
	
class Predict_SegNet:
	def __init__(self, outfold = "./folder/", model_iter=3):
		
		tf.reset_default_graph()
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(outfold + '/model/checkpoint') )
		
		#prep
		ckpt_raw = ckpt.model_checkpoint_path.strip().split("-")[:-1]
		ckpt_raw = "-".join(ckpt_raw)
		ckpt_raw = ckpt_raw + "-" + str(model_iter)
		#meta	
		ckpt_raw_meta = ckpt_raw + ".meta"
		imported_meta = tf.train.import_meta_graph(ckpt_raw_meta)
		

		self.sess = tf.Session()
		imported_meta.restore(self.sess, ckpt_raw)			
		graph = tf.get_default_graph()
		
		# ~ for op in graph.get_operations():
			# ~ print( op.name )
		# ~ input()
		self.X = graph.get_tensor_by_name('Model/X:0')
		self.Y = graph.get_tensor_by_name('Model/Y:0')
		self.logits = graph.get_tensor_by_name('Model/logits:0')
		self.logits_th = graph.get_tensor_by_name('Model/logits_th:0')
		self.iou_loss = graph.get_tensor_by_name('Model/loss:0')
		self.accu = graph.get_tensor_by_name('Model/Accuracy/Mean:0')
	
	
	def predict(self, im_batch):
		y, logits, logits_th, iou_loss, accu = self.sess.run([self.Y, self.logits, self.logits_th,self.iou_loss, self.accu], feed_dict={self.X:im_batch})
		return y, logits, logits_th, iou_loss, accu
		
	def close(self):
		self.sess.close()
		
	
if __name__ == "__main__":
	
	#Data- X_batch
	im = cv2.imread("/home/ai-nano/Documents/McMaster_box/test/test_resize/box_cti=1.85_cpi=2.692_lti=0.401_lpi=1.082_le=300000.png", 0)
	im = np.array(im).reshape(1,im.shape[0], im.shape[1], 1)

	#Model info
	model_fold = "/home/ai-nano/Documents/McMaster_box/Segmentation/"
	model_iter = 3
	manager = Predict_SegNet(model_fold, model_iter)

	#Run
	y, logits, logits_th, iou_loss, accu = manager.predict(im)
	print(y.shape)
	print(logits.shape)
	print(logits_th.shape)
	print(iou_loss)
	print(accu)
	
