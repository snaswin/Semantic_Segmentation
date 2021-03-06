import numpy as np
import tensorflow as tf
import pathlib
import os
import cv2
import matplotlib.pyplot as plt
import glob
import json

########################################################################
#	Prediction, Visualize prediction- Image batch
########################################################################

def allocate():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	return config

def write_json(jname,data):
	with open(jname, 'w') as outfile:
		json.dump(data, outfile, indent=3)


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
		

		self.sess = tf.Session( config=allocate() )
		imported_meta.restore(self.sess, ckpt_raw)			
		graph = tf.get_default_graph()
		
		# ~ for op in graph.get_operations():
			# ~ print( op.name )
		# ~ input()
		self.X = graph.get_tensor_by_name('Model/X:0')
		self.Y = graph.get_tensor_by_name('Model/Y:0')
		self.logits = graph.get_tensor_by_name('Model/predict:0')
		#self.logits_c1 = graph.get_tensor_by_name('Model/logits_c1:0')
		self.loss = graph.get_tensor_by_name('Model/loss:0')
		self.accu = graph.get_tensor_by_name('Model/Accuracy/Mean:0')
		
	
	#Predict
	def predict(self, x_batch, y_batch):
		# ~ y, logits, logits_c1, iou_loss, accu = self.sess.run([self.Y, self.logits, self.logits_c1,self.iou_loss, self.accu], feed_dict={self.X:x_batch, self.Y:y_batch})
		# ~ return y, logits, logits_c1, iou_loss, accu
		y, logits, loss, acc = self.sess.run([self.Y, self.logits, self.loss, self.accu], feed_dict={self.X:x_batch, self.Y:y_batch})
		return y, logits, loss, acc
		#Bug fix- accuracy
		
	# ~ def predict_real(self, x_batch):
		# ~ logits, logits_c1 = self.sess.run([self.logits, self.logits_c1], feed_dict={self.X:x_batch})
		# ~ return logits, logits_c1
	
	def predict_real(self, x_batch):
		predict = self.sess.run([self.logits], feed_dict={self.X:x_batch})
		return logits
	
	
	#Sample images
	def random_sample_imfiles(self,imfold="./folder/*.png", size=5):
		fnames = glob.glob(imfold)
		l = len(fnames)
		print("Len of imfold: ", l)
		inds = np.random.randint(0, l, size)
		fnames = np.take(fnames, inds)
		
		ims = []
		for fname in fnames:
			im = cv2.imread(fname,0)
			ims.append(im)
		return fnames, np.array(ims).reshape(-1, im.shape[0], im.shape[1], 1)
	
	#Sample im-pairs
	def random_sample_im_pairs(self,imfoldX="./folder/X/", imfoldY="./folder/Y/", fmt="png", size=5):
		fnames = glob.glob(imfoldX + "/*." + fmt)

		l = len(fnames)
		print("Len of imfold: ", l)
		inds = np.random.randint(0, l, size)
		fnames = np.take(fnames, inds)
		
		ims_X = []
		ims_Y = []
		names = []
		for fname in fnames:
			imX = cv2.imread(fname,0)
			ims_X.append(imX)
			
			name = fname.strip().split("/")[-1]
			names.append(name)
			
			imY = cv2.imread(imfoldY + "/" + name, 0)			
			ims_Y.append(imY)
			
		return names, np.array(ims_X).reshape(-1, imX.shape[0], imX.shape[1], 1), np.array(ims_Y).reshape(-1, imY.shape[0], imY.shape[1], 1)
	
	#Fragment Tensor & save images
	def fragment_save(self, tensor, fnames, outfold):
		pathlib.Path(outfold).mkdir(exist_ok=True, parents=True)
		for i,image in enumerate(tensor):
			name = outfold + "/" + fnames[i].strip().split("/")[-1]
			cv2.imwrite(name, image)
	
	def fragment_save_all(self, ims, y, logits, logits_c1, fnames, outfold, blended=True):
		#ims
		outfold_ims = outfold + "/ims/"
		self.fragment_save(ims, fnames, outfold_ims)
		#y
		outfold_y = outfold + "/y/"
		self.fragment_save(y*50, fnames, outfold_y)
		#logits
		outfold_logits = outfold + "/logits/"
		self.fragment_save(logits*50, fnames, outfold_logits)
		# ~ #logits_c1
		# ~ outfold_logits_c1 = outfold + "/logits_c1/"
		# ~ self.fragment_save(logits_c1*255, fnames, outfold_logits_c1)
		
		##BLENDED
		if blended == True:
			#Logits
			outfold_blend = outfold + "/blended_logits/"
			blends = []
			for i in range(ims.shape[0]):
				im1 = ims[i]
				im1 = im1.reshape(im1.shape[0], im1.shape[1])
				
				im2 = logits[i]
				im2 = np.array(im2, dtype=np.uint8)
				
				#out = self.blend(im1, im2, alpha=0.4,ch=0)
				out = self.blend_pixelwise(im1, im2, alpha=0.3, nclass_colors=[[0,0,0], [255,0,0], [255,153,255], [0,255,255] ])
				
				blends.append( out )
			blends = np.array(blends)
			self.fragment_save(blends, fnames, outfold_blend)
			
			# ~ #Logits_th
			# ~ outfold_blend = outfold + "/blended_logits_c1/"
			# ~ blends = []
			# ~ for i in range(ims.shape[0]):
				# ~ im1 = ims[i]
				# ~ im1 = im1.reshape(im1.shape[0], im1.shape[1])
				
				# ~ im2 = logits_c1[i]*255
				# ~ im2 = np.array(im2, dtype=np.uint8)
				
				# ~ out = self.blend(im1, im2, alpha=0.4, ch=0)
				
				# ~ blends.append( out )
			# ~ blends = np.array(blends)
			# ~ self.fragment_save(blends, fnames, outfold_blend)
			
			#y
			outfold_blend = outfold + "/blended_y/"
			blends = []
			for i in range(ims.shape[0]):
				im1 = ims[i]
				im1 = im1.reshape(im1.shape[0], im1.shape[1])
				
				im2 = y[i]
				im2 = np.array(im2, dtype=np.uint8)
				
				#out = self.blend(im1, im2, alpha=0.4, ch=2)
				out = self.blend_pixelwise(im1, im2, alpha=0.3, nclass_colors=[[0,0,0], [255,0,0], [255,153,255], [0,255,255] ])
				
				blends.append( out )
			blends = np.array(blends)
			self.fragment_save(blends, fnames, outfold_blend)
			#
	
	def blend(self, im1, im2, alpha=0.3, ch=0):
		#ch is the channel used to blend
		im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
		im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
		
		im2[:,:,ch] = 50
		return cv2.addWeighted(im1, alpha, im2, 1-alpha, 0.0)

	def blend_pixelwise(self, im1, im2_label, alpha=0.3, nclass_colors=[[0,0,0], [255,0,0], [255,153,255], [0,255,255] ]):
		
		#ch is the channel used to blend
		im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
		#im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
		im2 = np.zeros(im1.shape, dtype=im1.dtype)
		
		for i in range(im2_label.shape[0]):
			for j in range(im2_label.shape[1]):
				 l = int(im2_label[i][j])
				 im2[i][j][0] = nclass_colors[l][0]
				 im2[i][j][1] = nclass_colors[l][1]
				 im2[i][j][2] = nclass_colors[l][2]
				 
				 

		return cv2.addWeighted(im1, alpha, im2, 1-alpha, 0.0)

	
	def close(self):
		self.sess.close()


##====================##
### VISUALIZE & LOG ###
##====================##

def visualizer(manager, imfoldX, imfoldY, size, outfold, fmt="png"):
	#Fetch random test images
	names, ims_X, ims_Y = manager.random_sample_im_pairs(imfoldX, imfoldY, fmt, size)
	
	
	#Run-Predict
	y, logits, loss, accu = manager.predict(ims_X, ims_Y)
	print(y.shape)
	print(logits.shape)
	# ~ print(logits_c1.shape)
	print(loss)
	print(accu)
	
	print("#####\n####")
	ll = logits.ravel()
	print("Min and Max: ", ll.min(), ll.max())
	#plt.hist(ll, bins=16)
	#plt.savefig(outfold + "/predict_hist.png")
	#plt.show()
	
	# ~ fig, ax = plt.subplots(3)
	# ~ ax[0].imshow(y[0].reshape(logits[1].shape[:2]))
	# ~ ax[1].imshow(logits[1].reshape(logits[1].shape[:2]))
	# ~ ax[2].imshow(ims_Y[2].reshape(logits[1].shape[:2]))
	# ~ plt.show()
	
	# ~ input("waity")
	
	#Fragment & Save
	manager.fragment_save_all(ims_X, y, logits, None, names, outfold, blended=True)
	#Log iou_loss & accuracy
	jdata = {"loss": float(loss),
				"accuracy": float(accu),
				"imfoldX": imfoldX,
				"imfoldY": imfoldY,
				"test_ims": names,
				"Logits_min": ll.min(),
				"Logits_max" : ll.max() }
				
	write_json(outfold + "/predict_metrics.json",jdata)
	
	
	
if __name__ == "__main__":
	
	# ~ #Data- X_batch
	# ~ im = cv2.imread("/home/ai-nano/Documents/McMaster_box/test/test_resize/box_cti=1.85_cpi=2.692_lti=0.401_lpi=1.082_le=300000.png", 0)
	# ~ im = np.array(im).reshape(1,im.shape[0], im.shape[1], 1)

	## Model info
	#model_fold = "/data/McMaster/enclosure_2/enclosure-20-07-10-10-30-51_reformed_OUT/0/"
	model_fold = "/data/McMaster/real_data/real_reformed_OUT/3/"
	model_iter = 2
	manager = Predict_SegNet(model_fold, model_iter)

	
	## Visualize_args
	#imfoldX = "/data/McMaster/enclosure_2/enclosure-20-07-10-10-30-51_reformed/X/"
	#imfoldY = "/data/McMaster/enclosure_2/enclosure-20-07-10-10-30-51_reformed/Y/"
	imfoldX = "/data/McMaster/real_data/real_reformed/X/"
	imfoldY = "/data/McMaster/real_data/real_reformed/Y/"
	fmt = "png"
	
	size = 50
	outfold = model_fold + "/visualize/model-" + str(model_iter) + "/"

	#RUN Visualization
	visualizer(manager, imfoldX, imfoldY, size, outfold, fmt)
	
	
