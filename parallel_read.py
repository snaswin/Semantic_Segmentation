import numpy as np
import pathlib
import cv2
from numba import jit
import glob
from time import time

def names(path= "./folder/*.png"):
	fnames = glob.glob(path)
	return fnames

def read_images(fnames):
	out = []
	for name in fnames:
		out.append(cv2.imread(name,0))
	return np.array(out)

@jit(parallel=True)
def read_images_parallel(fnames):
	out = []
	for name in fnames:
		out.append(cv2.imread(name,0))
	return np.array(out)


if __name__ == "__main__":
	
	path = "/home/ai-nano/Documents/McMaster_box/test/test_resize_read/*.png"
	fnames = names(path)
	
	st = time()
	out = read_images(fnames)
	t1 = time()-st
	print("Serial: Time is ", t1)
	
	
	st = time()
	out = read_images_parallel(fnames)
	t1 = time()-st
	print("Parallel: Time is ", t1)
	
	
	
