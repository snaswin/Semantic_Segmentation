import numpy as np
import cv2
import glob
import os
import shutil
import pathlib
import matplotlib.pyplot as plt



def reform_datapairs(pathX, pathY, fmtX, fmtY, outfold):		
	outfoldx = outfold + "/X/"
	outfoldy = outfold + "/Y/"
	pathlib.Path(outfoldx).mkdir(exist_ok=True, parents=True)
	pathlib.Path(outfoldy).mkdir(exist_ok=True, parents=True)

	xpath = pathX + "/*." + fmtX
	ypath = pathY + "/*." + fmtY


	xfnames = glob.glob(xpath)
	for i, xname in enumerate(xfnames):
		print("# ", i, ", Working on ", xname)
		num = xname.strip().split("/")[-1].split(".")[0].split("-")[0]
		yname = pathY + "/" + num + "-inst." + fmtY
		
		if os.path.isfile(yname):
			#cp xname
			im = cv2.imread(xname)
			cv2.imwrite(outfoldx + "/" + num + ".png", im)
			#cp y
			im = cv2.imread(yname, 0)
			im = np.array(im/85, dtype=np.uint8)
			cv2.imwrite(outfoldy + "/" + num + ".png", im)

if __name__ == "__main__":
		
	pathX = "/data/McMaster/real_kmodel/glove_overhand_1_output/PNGImages/"
	pathY = "/data/McMaster/real_kmodel/glove_overhand_1_output/SegmentationClassPNG/"

	fmtX = "png"
	fmtY = "png"

	outfold = "/data/McMaster/real_kmodel/glove_overhand_1_output/"
	
	reform_datapairs(pathX, pathY, fmtX, fmtY, outfold)
	
	
