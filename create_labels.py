import numpy as np
import pathlib
import shutil
import cv2
import glob
import json
from tqdm import tqdm

def write_json(jname,data):
	with open(jname, 'w') as outfile:
		json.dump(data, outfile, indent=3)
	
def read_json(jname):
	with open(jname, 'r') as infile:
		data = json.load(infile)
	return data
	
if __name__ == "__main__":
	
	data_fold = "/data/McMaster/raw/"
	data_jname = "/data/McMaster/raw/label_readme.json"
	
	#train_data
	train_fold = "/data/McMaster/raw_ready/"
	
	d = read_json(data_jname)
	folders = list(d.keys())
	labels = list(d.values())
	
	j=0
	while j < len(folders):
		folders[j] = data_fold + "/" + folders[j]
		j = j+1
	
	xfold = train_fold + "/X/" 
	pathlib.Path(xfold).mkdir(exist_ok=True, parents=True)
	yfold = train_fold + "/Y/" 
	pathlib.Path(yfold).mkdir(exist_ok=True, parents=True)
	
	j=0
	while j < len(folders):
		folder = folders[j]
		label = labels[j]
		
		#handle folder
		fnames = glob.glob(folder + "/*.png")
		for fname in tqdm(fnames):
			name = fname.strip().split("/")[-1]
			y = np.array(cv2.imread(fname, 0) > 0).astype(np.uint8) * label
			
			#x copy
			shutil.copyfile(fname, xfold+"/"+name)
			#y write
			cv2.imwrite(yfold + "/" + name, y)
		j = j+1
		
		
			
