import glob
import cv2
import numpy as np
from numba import jit
from tqdm import tqdm
import pathlib

folder = "/home/ai-nano/Documents/McMaster_box/test/test/*.png"
outfold = "/home/ai-nano/Documents/McMaster_box/test/test_resize/"


fname = glob.glob(folder)
dsize= (512,512)

pathlib.Path(outfold).mkdir(exist_ok=True, parents=True)

for f in tqdm(fname):
	im = cv2.imread(f, 0)
	out = cv2.resize(im, dsize)
	outname = outfold + "/" + f.strip().split("/")[-1]
	cv2.imwrite(outname, out)
	
