import glob
import cv2
import numpy as np
from tqdm import tqdm
import pathlib
import os

# ~ mainfold = "/data/McMaster/real_data/"
# ~ xpath = mainfold + "Labeled_Dataset/*/PNGImages/*.png"
# ~ ypath = mainfold + "Labeled_Dataset/*/SegmentationClassPNG/*.png"
# ~ vispath = mainfold + "Labeled_Dataset/*/SegmentationClassVisualization/*.jpg"
mainfold = "/data/McMaster/remasked/"
xpath = mainfold + "/Occlusion Dataset/*/PNGImages/*.png"
ypath = mainfold + "/remasked_occlusion/*/SegmentationClassPNG/*.png"
vispath = mainfold + "/Occlusion Dataset/*/SegmentationClassVisualization/*.jpg"

outfold = "/data/McMaster/Combined/remasked_occlusion_full/"


xnames = sorted(glob.glob(xpath))
ynames = sorted(glob.glob(ypath))
vnames = sorted(glob.glob(vispath))

print("Total images sets: ", len(xnames))

xout = outfold + "/X/"
yout = outfold + "/Y/"
vout = outfold + "/Y_vis/"

pathlib.Path(xout).mkdir(exist_ok=True, parents=True)
pathlib.Path(yout).mkdir(exist_ok=True, parents=True)
pathlib.Path(vout).mkdir(exist_ok=True, parents=True)

for i, yname in enumerate(ynames):
	#base 38
	#screw 75
	#lid 113
	
	## Y
	im = cv2.imread(ynames[i], 0)
	im = im/84.0
	im = np.array(im, dtype=np.uint8)
	
	name = "{0:05}".format(i) + ".png"
	print(i, " Working on ", name)

	xn = xnames[i].strip().split("/")[-1].split(".")[0]
	yn = ynames[i].strip().split("/")[-1].split(".")[0]
	vn = vnames[i].strip().split("/")[-1].split(".")[0]
	
	if yn == xn and yn == vn:
		cv2.imwrite(yout+"/"+ name, im)
		
		## X
		im = cv2.imread(xnames[i])
		cv2.imwrite(xout+"/"+name, im)
		
		## V
		im = cv2.imread(vnames[i])
		cv2.imwrite(vout+"/"+name, im)
	else:
		print("At ", i)
		print("\t", xn)
		print("\t", yn)
		print("\t", vn)
		
