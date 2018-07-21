import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import math

from keras import __version__
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 1
BAT_SIZE = 32


def get_nb_files(directory):	#Get number of files by searching directory recursively

	if not os.path.exists(directory):
		return 0

	cnt = 0

	for r, dirs, files in os.walk(directory):
		for dr in dirs:
			cnt += len(glob.glob(os.path.join(r, dr + "/*")))

	return cnt


def train(args,model_load):

	nb_train_samples = get_nb_files(args.train_dir)
	nb_classes = len(glob.glob(args.train_dir + "/*"))
	nb_val_samples = get_nb_files(args.val_dir)
	nb_epoch = int(args.nb_epoch)
	batch_size = int(args.batch_size)

	train_datagen =  ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True
		)

	test_datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True
		)

	train_generator = train_datagen.flow_from_directory(
		args.train_dir,
		target_size=(IM_WIDTH, IM_HEIGHT),
		batch_size=batch_size,
		)

	validation_generator = test_datagen.flow_from_directory(
		args.val_dir,
		target_size=(IM_WIDTH, IM_HEIGHT),
		batch_size=batch_size,
		)

	model = load_model(model_load) 

	history_tl = model.fit_generator(
		train_generator,
		epochs=nb_epoch,
		steps_per_epoch=nb_train_samples//batch_size, 
		validation_data=validation_generator,
		validation_steps=nb_val_samples//batch_size, 
		class_weight='auto'
		)

	model.save(args.output_model_file)


if __name__=="__main__":

	a = argparse.ArgumentParser()
	a.add_argument("--train_dir", default='/home/aravind/DL/Data/train')
	a.add_argument("--val_dir", default='/home/aravind/DL/Data/test')
	a.add_argument("--nb_epoch", default=NB_EPOCHS)
	a.add_argument("--batch_size", default=BAT_SIZE)
	a.add_argument("--model_load", default='/home/aravind/DL/Model/inceptionv3-tl.model')
	a.add_argument("--output_model_file", default="/home/aravind/DL/Model/inceptionv3-tl-final.model")

	args = a.parse_args()

	if args.train_dir is None or args.val_dir is None:
		a.print_help()
		sys.exit(1)

	if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
		print("directories do not exist")
		sys.exit(1)

	train(args,model_load)

