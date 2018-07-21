""" Code to calculate no. of classes, no. of total images, max and min no. of images in a class """

import os, sys, glob

path = '/home/aravind/DL/Resized_images'
count = 0
lables = 0
max = 0
min = 200

for dir in os.listdir(path):

	lables+=1
	files = os.listdir(os.path.join(path,dir))
	count += len(files)
	if max < len(files):
		max = len(files)
		maxdir = dir
	if min > len(files):
		min = len(files)
		mindir = dir

print maxdir,max,mindir,min,lables,count
	
