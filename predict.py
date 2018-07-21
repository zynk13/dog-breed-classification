import sys
import os
import pickle
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input


target_size = (229, 229) #required size for Inception V3 model


def get_labels(pickle_file):

# Returns the list of all labels by reading from pickle file

	with open (pickle_file, 'rb') as fp:
    		labels = pickle.load(fp)
	return labels


def predict(model, img, target_size):

# Function to make predictions on input image given the model

	if img.size != target_size:
		img = img.resize(target_size)

	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)

	x = preprocess_input(x)		#Performs preprocessing required for Inception V3 model

	preds = model.predict(x)	#Makes predictions

	return preds[0]


if __name__=="__main__":
	
	dirname = os.path.dirname(__file__)

	a = argparse.ArgumentParser()
	a.add_argument("--image", help="Complete local path to the image")
	a.add_argument("--image_url", help="URL to image")
	a.add_argument("--model", default=os.path.join(dirname,"Model/inceptionv3-tl-final.model"))
	a.add_argument("--labels", default=os.path.join(dirname,"labels.pickle"))
	args = a.parse_args()

	if args.image is None and args.image_url is None:
		a.print_help()
		sys.exit(1)

	labels = get_labels(args.labels)

	model = load_model(args.model)

	if args.image is not None:				# If an image in directory is given as input

		img = Image.open(args.image)

		preds = predict(model, img, target_size)

		top_k = preds.argsort()[-5:][::-1]		# Get top 5 prediction score indices in sorted order

		print("")
		print("Directory Image Classification score")

		for node_id in top_k:
			label = labels[node_id]
			score = preds[node_id]
			print(str(label) + "	" +str(score))

	if args.image_url is not None:				# If an image URL is given as input

		response = requests.get(args.image_url)
		img = Image.open(BytesIO(response.content))

		preds = predict(model, img, target_size)

		top_k = preds.argsort()[-5:][::-1]		# Get top 5 prediction score indices in sorted order

		print("")
		print("Image URL Classification score")

		for node_id in top_k:
			label = labels[node_id]
			score = preds[node_id]
			print(str(label) + "	" +str(score))


