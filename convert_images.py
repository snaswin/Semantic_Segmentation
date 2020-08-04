import glob
import cv2
import numpy as np
from tqdm import tqdm
import pathlib
import os

main_folder = "Labeled Datast/"

subs = os.listdir(main_folder)
for sub in subs:
	print("Handling ", sub)
	folder = pathlib.Path(main_folder) / pathlib.Path(sub) / pathlib.Path("JPEGImages")
	fnames = os.listdir(folder)

	outfolder = main_folder + "/" + sub + "/PNGImages/"
	pathlib.Path(outfolder).mkdir(exist_ok=True, parents=True)

	for f in tqdm(fnames):
		name = str(folder) + "/" + f
		im = cv2.imread(name)
		outname = outfolder + "/" + f.strip().split(".")[0] + ".png"
		cv2.imwrite(outname, im)
		
		
