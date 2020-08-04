import glob
import cv2
import numpy as np
from tqdm import tqdm
import pathlib
import os

mainfold = "/data/McMaster/real_data/"
xpath = mainfold + "Labeled_Dataset/*/PNGImages/*.png"
ypath = mainfold + "Labeled_Dataset/*/SegmentationClassPNG/*.png"
vispath = mainfold + "Labeled_Dataset/*/SegmentationClassVisualization/*.jpg"

outfold = "/data/McMaster/real_data/real_reformed/"

xnames = glob.glob(xpath)
ynames = glob.glob(ypath)
vnames = glob.glob(vispath)

print("Total images sets: ", len(xnames))

xout = outfold + "/X/"
yout = outfold + "/Y_vis/"
vout = outfold + "/Y_overlap/"

pathlib.Path(xout).mkdir(exist_ok=True, parents=True)
pathlib.Path(yout).mkdir(exist_ok=True, parents=True)
pathlib.Path(vout).mkdir(exist_ok=True, parents=True)

for i, yname in enumerate(ynames):
	#base 38
	#screw 75
	#lid 113
	
	## Y
	im = cv2.imread(ynames[i], 0)
	im = im/35
	im = np.array(im, dtype=np.uint8)
	for ii in range(im.shape[0]):
		for jj in range(im.shape[1]):
			if abs(im[ii][jj] - 2.0) <0.001:
				im[ii][jj] = 3
			elif abs(im[ii][jj] - 3.0) <0.001:
				im[ii][jj] = 2
			else:
				pass
	im = im * 35
	
	name = "{0:05}".format(i) + ".png"
	print(i, " Working on ", name)
	cv2.imwrite(yout+"/"+ name, im)
	
	## X
	im = cv2.imread(xnames[i])
	cv2.imwrite(xout+"/"+name, im)
	
	## V
	im = cv2.imread(vnames[i])
	cv2.imwrite(vout+"/"+name, im)

