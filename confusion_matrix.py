#Program that calculates a confution matrix for a given neural network and testset

#imports
#-----------------------------------------------------------------------
import tensorflow as tf
import argparse
import numpy as np
import pathlib
import shutil
import sys
import os

from tensorflow.python.platform import gfile
from PIL import Image






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

BATCH_SIZE 		= FLAGS.batch_size
IMG_WIDTH 		= None
IMG_HEIGHT		= None
IMAGE_COUNT 		= 0




#-----------------------------------------------------------------------
def load_sample_metadata():
	global CLASS_NAMES
	global IMAGE_COUNT

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
	print("Total number of samples in testset: "+str(IMAGE_COUNT))


	print("-------------------------------------------------")
	
	
  
#-----------------------------------------------------------------------
if FLAGS.arch == 'resnet':
	IMG_WIDTH 	= 224
	IMG_HEIGHT	= 224
else:
	IMG_WIDTH 	= 299
	IMG_HEIGHT	= 299

load_sample_metadata()



nn_model = tf.keras.models.load_model(FLAGS.model_file)
print("Neural network model loaded.")
nn_model.summary()


#create matrix with dict
#id = (actual, predicted), value = count
confusion_matrix = {}
for actual_class in CLASS_NAMES:
	for predicted_class in CLASS_NAMES:
		confusion_matrix[ (actual_class, predicted_class) ] = 0 #init count at 0


data_dir = pathlib.Path(FLAGS.test_dir)
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
for img_file in list_ds:
	img = tf.io.read_file( img_file )
	img = tf.image.decode_jpeg(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)
	img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
	img = np.expand_dims(img, axis=0)
	img = img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)

	prediction = nn_model.predict(img)
	#--------------------------------------------------------
		
	img_path = img_file.numpy()
	dest_path = os.path.split(img_path)
	parent_dir = os.path.split(dest_path[0])

	#Add result to confusion matrix
	predicted_class = CLASS_NAMES[ np.argmax(prediction[0]) ]
	actual_class = parent_dir[1].decode()
	
	confusion_matrix[ (actual_class, predicted_class) ] = 1 + confusion_matrix[ (actual_class, predicted_class) ]
	print("Actual class: "+actual_class+", Predicted class: "+predicted_class+", Matrix value "+str(confusion_matrix[ (actual_class, predicted_class) ]))

print(confusion_matrix)

input("Fin")
