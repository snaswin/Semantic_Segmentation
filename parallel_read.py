import numpy as np
import pathlib
import cv2
import glob
from time import time
#from scipy.misc import imread
from sklearn.externals.joblib import Parallel, delayed


def names(path= "./folder/*.png"):
	fnames = glob.glob(path)
	return fnames

def read_images(fnames):
	out = []
	for name in fnames:
		out.append(cv2.imread(name,0))
	return np.array(out)


def read_images_parallel(fnames):
	images = Parallel(n_jobs=4, verbose=5)( delayed(cv2.imread)(f) for f in fnames)
	return images

if __name__ == "__main__":
	
	path = "/home/ai-nano/Documents/McMaster_box/test/test_resize_read/*.png"
	fnames = names(path)
	
	st = time()
	out = read_images(fnames)
	print("OUT: ", out.shape)
	t1 = time()-st
	print("Serial: Time is ", t1)
	
	
	st = time()
	out = read_images_parallel(fnames)
	t1 = time()-st
	print("Parallel: Time is ", t1)
	
	
	
