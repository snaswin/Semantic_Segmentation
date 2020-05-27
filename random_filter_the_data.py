import glob
import cv2
import numpy as np
from numba import jit
from tqdm import tqdm
import pathlib

folder = "/home/aswin-rpi/Documents/GITs/test_resize/*.png"

fnames = glob.glob(folder)
print(fnames)
