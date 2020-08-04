import glob
import cv2
import numpy as np
from tqdm import tqdm
import pathlib
import os


ypath = "/data/McMaster/real_data/real_reformed/Y_vis/*.png"
yout = "/data/McMaster/real_data/real_reformed/Y/"

ynames = glob.glob(ypath)
pathlib.Path(yout).mkdir(exist_ok=True, parents=True)

for i, yname in enumerate(ynames):
	im = cv2.imread(ynames[i], 0)
	im = im / 35
	im = np.array(im, dtype=np.uint8)
	
	name = yname.strip().split("/")[-1]
	print(i, " Working on ", name)
	cv2.imwrite(yout+"/"+ name, im)
