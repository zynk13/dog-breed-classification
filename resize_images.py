""" Code to resize images to the 299x299 dimension required for the Inception V3 model. Resizing done by adding black fillers unlike keras' default resize which squishes the image """

import os, sys, glob
from PIL import Image

size = (299, 299)
filetypes =['*.jpg','*.jpeg']
 
os.mkdir('/home/aravind/DL/Resized_images')

outpath = '/home/aravind/DL/Resized_images'
inpath = '/home/aravind/DL/Augmented_images'

for dir in os.listdir(inpath):

	os.mkdir(os.path.join(outpath,dir))

	src = os.path.join(inpath,dir)
	dest = os.path.join(outpath,dir)

	for file in os.listdir(src):
		infile = os.path.join(src,file)
		outfile = os.path.join(dest,file)

		try:
		    im = Image.open(infile)
		    im.thumbnail(size, Image.ANTIALIAS)
		    old_im_size = im.size
		    
		    new_im = Image.new("RGB", size)

		    new_im.paste(im, ((size[0]-old_im_size[0])/2,
				      (size[1]-old_im_size[1])/2))
		    
		    new_im.save(outfile, "JPEG")

		except IOError:
		    print "Cannot resize '%s'" % infile
