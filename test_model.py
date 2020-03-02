#Program that loads a trained model and evaluates its accuracy on a 
#test set


#imports
#-----------------------------------------------------------------------
import tensorflow as tf
import argparse
import numpy as np
import pathlib
import sys
import os

from tensorflow.python.platform import gfile




#command-line parameters
#-----------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--test_dir', type=str,required=True,
	help='Folder containing the test set'
)

parser.add_argument('--model_file', type=str,required=True,
	help='HDF5 file'
)

parser.add_argument('--arch', type=str,required=True,
	help='Neural network architecture: <resnet|googlenet>'
)

parser.add_argument('--batch_size', type=int,required=False,
	default=32,
	help='Number of images per batch'
)

parser.add_argument('--mode', type=str,required=False,
	default='all',
	help='Print <all|global|detailed> test'
)

FLAGS, unparsed = parser.parse_known_args()
	

#param validation
#-----------------------------------------------------------------------
if FLAGS.arch not in ('resnet', 'googlenet'):
	print("Error: 'arch' must be 'resnet' or 'googlenet'", 
			file=sys.stderr)
	quit()

if FLAGS.batch_size <= 0:
	print("Error: 'batch_size' must be greater than 0", 
			file=sys.stderr)
	quit()



#global variables
#-----------------------------------------------------------------------
CLASS_NAMES 		= None

BATCH_SIZE 			= FLAGS.batch_size
SHUFFLE_BUFFER_SIZE = 1000
IMG_WIDTH 			= None
IMG_HEIGHT			= None
IMAGE_COUNT 		= 0
STEPS_PER_EPOCH 	= 0


AUTOTUNE 			= tf.data.experimental.AUTOTUNE




#-----------------------------------------------------------------------
def get_label(file_path):
	# convert the path to a list of path components
	parts = tf.strings.split(file_path, '/')
	# The second to last is the class-directory
	return parts[-2] == CLASS_NAMES
 
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
  
def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


#-----------------------------------------------------------------------
def load_sample_metadata():
	global CLASS_NAMES
	global IMAGE_COUNT
	global VAL_IMG_COUNT
	global TEST_IMG_COUNT
	global STEPS_PER_EPOCH
	global VAL_STEPS_PER_EPOCH
	global TEST_STEPS

	imgset = FLAGS.test_dir
		
	
	print("Loading sample metadata...")
	
	if not gfile.Exists(imgset):
		print("Image directory '" + imgset + "' not found.", 
				file=sys.stderr)
		quit()

	data_dir = pathlib.Path(imgset)	

	CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
	CLASS_NAMES =np.sort(CLASS_NAMES)
	print("Labels detected: "+str(CLASS_NAMES))
	
	IMAGE_COUNT = len(list(data_dir.glob('*/*.jpg') )
						+list(data_dir.glob('*/*.JPG'))
						+list(data_dir.glob('*/*.jpeg'))
						+list(data_dir.glob('*/*.JPEG')) )
	print("Total number of samples in trainset: "+str(IMAGE_COUNT))
	
	STEPS_PER_EPOCH = np.ceil( IMAGE_COUNT/BATCH_SIZE )
	print("Steps per epoch: "+str(STEPS_PER_EPOCH))


	print("-------------------------------------------------")
		
	
	
#-----------------------------------------------------------------------	
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, 
							is_test=False):
	# use `.cache(filename)` to cache preprocessing work for datasets that 
	#don't fit in memory.
	if cache:
		if isinstance(cache, str):
			ds = ds.cache(cache)
		else:
			ds = ds.cache()
	
	if is_test == False:
		#train or val
		ds = ds.shuffle(buffer_size=shuffle_buffer_size)
		
		# Repeat forever
		ds = ds.repeat()
	
	ds = ds.batch(BATCH_SIZE)

	# `prefetch` lets the dataset fetch batches in the background while
	# the model is training.
	ds = ds.prefetch(buffer_size=AUTOTUNE)
	
	return ds
	
  
	
#-----------------------------------------------------------------------
def load_data(imgset):
	data_dir = pathlib.Path(imgset)
	list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
		
	# Set `num_parallel_calls` so multiple images are loaded/processed 
	# in parallel.
	labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
	
	return labeled_ds
	
	
  
#-----------------------------------------------------------------------
if FLAGS.arch == 'resnet':
	IMG_WIDTH 	= 224
	IMG_HEIGHT	= 224
else:
	IMG_WIDTH 	= 299
	IMG_HEIGHT	= 299

load_sample_metadata()

test_dataset	= load_data(FLAGS.test_dir)
test_dataset	= prepare_for_training(test_dataset, "cached_test", 
										SHUFFLE_BUFFER_SIZE, is_test=True)



nn_model = tf.keras.models.load_model(FLAGS.model_file)
print("Neural network model created.")
nn_model.summary()

if FLAGS.mode == 'all' or FLAGS.mode == 'global':
	print("Global accuracy")
	nn_model.evaluate(test_dataset)

if FLAGS.mode == 'all' or FLAGS.mode == 'detailed':
	print("Detailed accuracy")
	data_dir = pathlib.Path(FLAGS.test_dir)
	list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
	for img_file in list_ds:
		img = tf.io.read_file( img_file )
		img = tf.image.decode_jpeg(img, channels=3)
		img = tf.image.convert_image_dtype(img, tf.float32)
		img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
		img = np.expand_dims(img, axis=0)
		img = img.reshape(1, IMG_WIDTH, IMG_WIDTH, 3)

		prediction = nn_model.predict_classes(img)
		probabilities = nn_model.predict(img)
		i = 0
		img_probs = []
		order_probs = {}
		for p in probabilities[0]:
			img_probs.append( str(CLASS_NAMES[i])+ ":" + str(round(p*100, 7)) + "%" )
			order_probs[CLASS_NAMES[i]]=(round(p*100, 3))
			i = i + 1
		order_probs = sorted( ((v,k) for k,v in order_probs.items()), reverse=True)
		print(img_file)
		print( "Model prediction: " + CLASS_NAMES[ prediction[0] ]+" - probabilities: "+str(img_probs)+" - top: "+str(order_probs))
		print("------------------------------------------------------")
	
	


input("Fin")
