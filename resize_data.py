import glob
import cv2
import numpy as np
from numba import jit
from tqdm import tqdm
import pathlib

folder = "/data/McMaster/raw_ready/**/*.png"
outfold = "/data/McMaster/raw_ready_resize/"


fname = glob.glob(folder)[260000+244971:]
dsize= (512,512)

pathlib.Path(outfold + "/X/").mkdir(exist_ok=True, parents=True)
pathlib.Path(outfold + "/Y/").mkdir(exist_ok=True, parents=True)


for f in tqdm(fname):
	im = cv2.imread(f, 0)
	out = cv2.resize(im, dsize)
	outname = outfold + "/" + "/".join(f.strip().split("/")[-2:] )
	cv2.imwrite(outname, out)
	
