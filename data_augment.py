""" File that creates augmented images to make the total number of samples in each class balanced and equal"""

import os, sys, glob
import math
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Data generator that generates variations of input data by rotation, zoom, shear, etc.

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Creates the destination directory

os.mkdir('/home/aravind/DL/Augmented_images')

# Variables for source and destination directories

inpath = '/home/aravind/DL/Dog_breed_dataset'
outpath = '/home/aravind/DL/Augmented_images'

# The amount of images required in every class

class_size = 150

# Following code recreates source directory structure in the destination 
# and also creates extra images to satisfy the required class size by data augmentation

for dir in os.listdir(inpath):

	os.mkdir(os.path.join(outpath,dir))

	count = len(os.listdir(os.path.join(inpath,dir)))
	ratio=math.floor(class_size/count)-1
	if ratio <= 1:
		ratio = 2
	curr_dir = os.path.join(inpath,dir)
	dest_dir = os.path.join(outpath,dir)
	count = 0
	for file in os.listdir(curr_dir):
		img=load_img(os.path.join(curr_dir,file))
		img.save(os.path.join(dest_dir,file))
		x=img_to_array(img) 
		x=x.reshape((1,) + x.shape)
		i=0
		for batch in datagen.flow(x, batch_size=1,save_to_dir=dest_dir, save_format='jpg'):
			i+=1
			if i > ratio:
				count+=i+1
				break
		if count >= class_size:
			break

