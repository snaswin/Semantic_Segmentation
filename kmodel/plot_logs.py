
import os
import glob
import numpy as np
import random
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

#mainfold = "/data/McMaster/enclosure_2/enclosure-20-07-10-10-30-51_reformed_Kout_small/0/"
#mainfold = "/data/McMaster/real_kmodel/glove_overhand_1_output_Kout_small/0/"
#mainfold = "/data/McMaster/real_data_full/real_full_reformed_Ksmall_full/2/"
mainfold = "/data/McMaster/Combined/Output/comb_Kout_small/0/"
save = True

fname = mainfold + "/log.csv"
df = pd.read_csv(fname, delimiter=",")

epochs = df["epoch"]
train_acc = df["accuracy"]
val_acc = df["val_accuracy"]

train_loss = df["loss"]
val_loss = df["val_loss"]

plt.plot(epochs, train_acc, label="train_acc")
plt.plot(epochs, val_acc, label="valid_acc")
plt.legend()
if save == True:
	plt.savefig(mainfold + "/accuracy.png")
	plt.clf()
else:
	plt.show()

plt.plot(epochs, train_loss, label="train_loss")
plt.plot(epochs, val_loss, label="val_loss")
plt.legend()
if save == True:
	plt.savefig(mainfold + "/loss.png")
	plt.clf()
else:
	plt.show()
