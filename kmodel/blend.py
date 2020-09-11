import numpy as np
import pathlib
import os
import cv2
import matplotlib.pyplot as plt
import glob
import json

def blend(im1, im2, alpha=0.5):
	#ch is the channel used to blend
	print(im1.shape)
	
	# ~ im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
	# ~ im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
	
	return cv2.addWeighted(im1, alpha, im2, 1-alpha, 0.0)


mainfold = "/data/McMaster/Combined/Output/comb_Kout_small/0/visualize_real/66/"

a = mainfold + "/imx/*.png"
b = mainfold + "/imyhat/*.png"
outfold = mainfold + "/overlap/"

pathlib.Path(outfold).mkdir(exist_ok=True, parents=True)

fa = glob.glob(a)
fb = glob.glob(b)

#c = 0
for i,ff in enumerate(fa):
	aa = cv2.imread(fa[i])
	bb = cv2.imread(fb[i])
	out = blend(aa, bb, alpha=0.5)
	
	name = fa[i].strip().split("/")[-1]
	cv2.imwrite(outfold+ "/" + name, out)
	#c = c+1
	
	
