""" Code to split the dataset into train and test folders randomly with 80% of the data in train """

import os
import shutil
import numpy as np

os.mkdir('/home/aravind/DL/Data')
os.mkdir('/home/aravind/DL/Data/train')
os.mkdir('/home/aravind/DL/Data/test')

src = '/home/aravind/DL/Resized_images'
dest_train = '/home/aravind/DL/Data/train'
dest_test = '/home/aravind/DL/Data/test'

for dir in os.listdir(src):
	
	cur_src = os.path.join(src,dir)
	cur_train = os.path.join(dest_train,dir)
	cur_test = os.path.join(dest_test,dir)

	if not os.path.exists(cur_train):
	    os.makedirs(cur_train)
	if not os.path.exists(cur_test):
	    os.makedirs(cur_test)

	files = os.listdir(cur_src)

	for f in files:
		if np.random.rand(1) < 0.2:
			shutil.move(os.path.join(cur_src,f) , os.path.join(cur_test,f))
		else:
			shutil.move(os.path.join(cur_src,f) , os.path.join(cur_train,f))
	


