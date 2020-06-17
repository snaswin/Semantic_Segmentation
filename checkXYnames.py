import glob
import os


pathx = "/data/McMaster/raw_ready_resize/X/"
pathy = "/data/McMaster/raw_ready_resize/Y/"

x_names = os.listdir(pathx)
y_names = os.listdir(pathy)

# ~ print(len(x_names))
# ~ print(len(y_names))


C= 0
j = 0
while j<len(x_names):
	
	n1 = x_names[j]
	n2 = y_names[j]
	
	if n1 == n2:
		C +=1
	else:
		print("Not a match at ", j)
	
	j = j+1

print("Count is ", C)	

