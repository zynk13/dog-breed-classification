# dog-breed-classification
Transfer learn the Inception V3 CNN to classify input images to their respective dog breeds among the possible 133

predict.py :

	- Used to make preditions on the input image, classifying it into one of the 133 dog breeds.
	- Argument --image used to give path to input image for classification
	- Argument --image_url used to give url to input image for classification
	- Takes model file from /Model by default, please input --model otherwise.
	- Reads labels from labels.pickle file by default, please input --labels for other paths.
	- example usage : python predict.py --image_url=https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/
	uploads/2017/11/12224133/Silky-Terrier-On-White-01.jpg 

Data :	

	-Contains both the training and test data sets in appropriately named folders. 
	-Several steps of pre-processing performed on the original dataset to obtain contents. 
	-Preserved directory structure with folder names as labels but image name format not preserved in extra images.

Preprocessing scripts :
	
	data_augment.py - Used to create augmented images to make the total number of images in each class balanced and equal.

	resize_images.py - Used to resize images to 299x299 which is the required dimension for Inception V3 model.
			 - Resizing done by adding black fillers unlike keras' default resize which squishes the image.

	split_train_test.py - Used to split the dataset into train and test folders randomly with 80% of the data in train.


Training Code :

	train.py - Used to train the model with transfer learning from Inception V3 model.
				- Uses images from --train_dir and --test_dir passed as arguments to train and test the model respectively.
				- Considers the sub-folder names as classes to classify image into (dog breeds).
				- Saves the compiled and trained model in a model file mentioned in the argument --output_model_file.

	retrain.py - Used to perform additional training in a partially trained model file.
				  - Uses the model file in given argument --model_load.
				  - Saves the compiled and trained model in a model file mentioned in the argument --output_model_file.

  
labels.pickle :
	
	Contains all the class/label names in order 
