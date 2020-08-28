
import os
import glob

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import random
import matplotlib.pyplot as plt
import pathlib


#Prep paths
#main_dir = "/data/McMaster/enclosure_2/enclosure-20-07-10-10-30-51_reformed/"
main_dir = "/data/McMaster/real_kmodel/glove_overhand_1_output/"
input_dir = main_dir + "/X/"
target_dir = main_dir + "/Y/"

#chpt
#check_dir = "/data/McMaster/enclosure_2/enclosure-20-07-10-10-30-51_reformed_Kout/2/"
check_dir = "/data/McMaster/real_kmodel/glove_overhand_1_output_Kout/0/"
epoch = "01"

checkpoint_dir = check_dir + "/model/"
latest = checkpoint_dir + "/synth_segmentation-" + epoch +".hdf5"
print("\n\nLatest is ", latest)

#outfold
outfold = check_dir + "/visualize/" + epoch + "/"
pathlib.Path(outfold).mkdir(exist_ok=True, parents=True)

img_size = (512, 512)
num_classes = 4
batch_size = 32


input_img_paths = sorted( glob.glob(input_dir + "/*.png") )
target_img_paths = sorted( glob.glob(target_dir + "/*.png") )

print("Num of ims- ", len(input_img_paths))


class Data_handler(tf.keras.utils.Sequence):
	"Iter over data"
	
	def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
		self.batch_size = batch_size
		self.img_size = img_size
		self.input_img_paths = input_img_paths
		self.target_img_paths = target_img_paths
	
	#num of batches
	def __len__(self):
		return len(self.target_img_paths) // self.batch_size
	
	def __getitem__(self, idx):
		#return tuple (input, target) corresponds to batch #idx
		b_i = idx * self.batch_size
		
		batch_xnames = self.input_img_paths[b_i : b_i+self.batch_size]
		batch_ynames = self.target_img_paths[b_i : b_i+self.batch_size]
		
		batch_x = np.zeros( (self.batch_size,)+self.img_size+(1,) , dtype="float32")
		for xx, xpath in enumerate(batch_xnames):
			im = load_img(xpath, target_size=self.img_size, color_mode="grayscale")
			im = np.expand_dims(im, 2)
			batch_x[xx] = im
		
		batch_y = np.zeros( (self.batch_size,)+self.img_size+(1,) , dtype="uint8")
		for yy, ypath in enumerate(batch_ynames):
			im = load_img(ypath, target_size=self.img_size, color_mode="grayscale")
			im = np.expand_dims(im, 2)
			batch_y[yy] = im
			
		return np.asarray(batch_x), np.asarray(batch_y)
		
def get_model(img_size, num_class):
	inputs = tf.keras.Input( shape=img_size + (1,) )
	
	#encoder
	x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation("relu")(x)
	
	previous_block_activation = x  # Set aside residual
	
	#Blocks 1,2,3 except the filters
	
	for filters in [64,128,256]:
		x = tf.keras.layers.Activation("relu")(x)
		x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
		x = tf.keras.layers.BatchNormalization()(x)
		
		x = tf.keras.layers.Activation("relu")(x)
		x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
		x = tf.keras.layers.BatchNormalization()(x)
		
		x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
		
		#residual
		residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
		x = tf.keras.layers.add([x, residual])
		
		previous_block_activation = x	#Set aside residual
		
	#decoder
	for filters in [256,128,64,32]:
		x = tf.keras.layers.Activation("relu")(x)
		x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
		x = tf.keras.layers.BatchNormalization()(x)
		
		x = tf.keras.layers.Activation("relu")(x)
		x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
		x = tf.keras.layers.BatchNormalization()(x)
		
		x = tf.keras.layers.UpSampling2D(2)(x)
		
		#residual
		residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
		residual = tf.keras.layers.Conv2D(filters, 1, padding="same")(residual)
		x = tf.keras.layers.add([x, residual])
		previous_block_activation = x
		
	
	#Pixelwise classification
	outputs = tf.keras.layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
	
	#Define
	model = tf.keras.Model(inputs, outputs)
	return model

#Validation split
val_samples = 50
random.Random(7).shuffle(input_img_paths)
random.Random(7).shuffle(target_img_paths)

train_xpath = input_img_paths[: -val_samples]
train_ypath = target_img_paths[: -val_samples]

val_xpath = input_img_paths[-val_samples:]
val_ypath = target_img_paths[-val_samples:]

#instantiate data handler
train_handle = Data_handler(batch_size, img_size, train_xpath, train_ypath)
val_handle = Data_handler(batch_size, img_size, val_xpath, val_ypath)

### Model
# ~ latest = tf.train.latest_checkpoint(checkpoint_dir)
# ~ print("\n\nLatest checkpoint is ", latest)
# ~ latest = "/home/ai-nano/Documents/Segmentation_keras/synth_segmentation-01-0.53.hdf5"
# ~ print("\n\nLatest is ", latest)

#tf reset graph
tf.keras.backend.clear_session()

#build
model = get_model(img_size, num_classes)
model.load_weights(latest)
model.summary()

#Compile graph
opt = tf.keras.optimizers.RMSprop(learning_rate=1e-2)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics="accuracy")

# ~ #eval
# ~ test_images, test_labels = val_handle.__getitem__(1)
# ~ loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
# ~ print("Validation results - ", loss, acc)


def display_pred(val_pred, i):
	pred = np.argmax( val_pred[i], axis=-1)
	pred = np.expand_dims(pred, axis=-1)
	im = tf.keras.preprocessing.image.array_to_img(pred)
	return im

def output_pred(val_pred):
	pred = np.argmax(val_pred, axis=-1)
	pred = np.expand_dims(pred, axis=-1)
	return pred

#predict
batch = 0
imx, imy = train_handle.__getitem__(batch)
val_pred = model.predict(imx)
imyhat = output_pred(val_pred)
print(imyhat.shape)

# ~ n = 1
# ~ fig, ax = plt.subplots(3)
# ~ ax[0].imshow(imx[n][:,:,0])
# ~ ax[1].imshow(imyhat[n][:,:,0])
# ~ ax[2].imshow(imy[n][:,:,0])
# ~ plt.show()

ns = imx.shape[0]
imx_fold = outfold + "/imx/"
imyhat_fold = outfold + "/imyhat/"
imy_fold = outfold + "/imy_fold/"
pathlib.Path(imx_fold).mkdir(exist_ok=True, parents=True)
pathlib.Path(imyhat_fold).mkdir(exist_ok=True, parents=True)
pathlib.Path(imy_fold).mkdir(exist_ok=True, parents=True)
for n in range(ns):
	num = batch*batch_size + n
	plt.imsave(imx_fold + "/" + str(num) + ".png", imx[n][:,:,0] )
	plt.imsave(imyhat_fold + "/" + str(num) + ".png", imyhat[n][:,:,0] )
	plt.imsave(imy_fold + "/" + str(num) + ".png", imy[n][:,:,0] )
	


# ~ im = display_pred(val_pred, 0)
# ~ plt.imshow(im)
# ~ plt.show()

