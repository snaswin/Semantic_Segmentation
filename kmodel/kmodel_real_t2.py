
import os
import glob

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import random
import pathlib
from keras.callbacks import CSVLogger

#Prep paths
#main_dir = "/data/McMaster/enclosure_2/enclosure-20-07-10-10-30-51_reformed/"
main_dir = "/data/McMaster/real_kmodel/glove_overhand_1_output/"

input_dir = main_dir + "/X/"
target_dir = main_dir + "/Y/"
fetch_size = 338

outfold = "/data/McMaster/real_kmodel/glove_overhand_1_output_Kout_small/"
pathlib.Path(outfold).mkdir(exist_ok=True, parents=True)
outfold = outfold + "/" + str(len(os.listdir(outfold))) +"/"


img_size = (512, 512)
num_classes = 4
batch_size = 32

input_img_paths = sorted( glob.glob(input_dir + "/*.png") )[:fetch_size]
target_img_paths = sorted( glob.glob(target_dir + "/*.png") )[:fetch_size]

print("Num of ims- ", len(input_img_paths))

for ipath, tpath in zip(input_img_paths[:5], target_img_paths[:5]):
	print(ipath, " & ", tpath)


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
		

#MODEL
def get_model(img_size, num_class):
	inputs = tf.keras.Input( shape=img_size + (1,) )
	
	#encoder
	x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation("relu")(x)
	
	previous_block_activation = x  # Set aside residual
	
	#Blocks 1,2,3 except the filters
	
	for filters in [8,16,32]:
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
	for filters in [32, 16, 8, 4]:
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

#tf reset graph
tf.keras.backend.clear_session()

#build
model = get_model(img_size, num_classes)
model.summary()


#Validation split
val_samples = int(0.1 * fetch_size)
random.Random(7).shuffle(input_img_paths)
random.Random(7).shuffle(target_img_paths)

train_xpath = input_img_paths[: -val_samples]
train_ypath = target_img_paths[: -val_samples]

val_xpath = input_img_paths[-val_samples:]
val_ypath = target_img_paths[-val_samples:]

#instantiate data handler
train_handle = Data_handler(batch_size, img_size, train_xpath, train_ypath)
val_handle = Data_handler(batch_size, img_size, val_xpath, val_ypath)

#Compile graph
opt = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics="accuracy")

model_fold = outfold + "/model/"
pathlib.Path(model_fold).mkdir(exist_ok=True, parents=True)

# ~ filepath= "synth_segmentation-{epoch:02d}-{val_accuracy:.2f}.hdf5"
filepath= model_fold + "synth_segmentation-{epoch:02d}.hdf5"
csv_logger = CSVLogger(outfold + '/log.csv', append=True, separator=',')

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath, 
				monitor='val_loss', 
				verbose=0, 
				save_best_only=False, 
				save_weights_only=False, 
				mode='auto', 
				period=1), csv_logger]

#Start training
epochs = 150
model.fit(train_handle, epochs=epochs, validation_data = val_handle, callbacks=callbacks)

