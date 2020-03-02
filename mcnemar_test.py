#Program that calculates the mcnemar test for 2 networks

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
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

from statsmodels.stats.contingency_tables import mcnemar




#command-line parameters
#-----------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--test_dir', type=str,required=True,
	help='Folder containing the test set'
)

parser.add_argument('--model_1_file', type=str,required=True,
	help='HDF5 file'
)

parser.add_argument('--model_2_file', type=str,required=True,
	help='HDF5 file'
)

parser.add_argument('--arch_1', type=str,required=True,
	help='Neural network architecture: <resnet|googlenet>'
)

parser.add_argument('--arch_2', type=str,required=True,
	help='Neural network architecture: <resnet|googlenet>'
)

parser.add_argument('--batch_size', type=int,required=False,
	default=32,
	help='Number of images per batch'
)


FLAGS, unparsed = parser.parse_known_args()
	

#param validation
#-----------------------------------------------------------------------
if FLAGS.arch_1 not in ('resnet', 'googlenet'):
	print("Error: 'arch_1' must be 'resnet' or 'googlenet'", 
			file=sys.stderr)
	quit()

if FLAGS.arch_2 not in ('resnet', 'googlenet'):
	print("Error: 'arch_2' must be 'resnet' or 'googlenet'", 
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
IMG_WIDTH_1 		= None
IMG_HEIGHT_1		= None
IMG_WIDTH_2 		= None
IMG_HEIGHT_2		= None
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
if FLAGS.arch_1 == 'resnet':
	IMG_WIDTH_1 	= 224
	IMG_HEIGHT_1	= 224
else:
	IMG_WIDTH_1 	= 299
	IMG_HEIGHT_1	= 299


if FLAGS.arch_2 == 'resnet':
	IMG_WIDTH_2 	= 224
	IMG_HEIGHT_2	= 224
else:
	IMG_WIDTH_2		= 299
	IMG_HEIGHT_2	= 299

load_sample_metadata()



nn_model_1 = tf.keras.models.load_model(FLAGS.model_1_file)
print("Neural network model loaded.")
nn_model_1.summary()

nn_model_2 = tf.keras.models.load_model(FLAGS.model_2_file)
print("Neural network model loaded.")
nn_model_2.summary()


data_dir = pathlib.Path(FLAGS.test_dir)
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
results = []
yes_yes = 0
yes_no = 0
no_yes = 0
no_no = 0
for img_file in list_ds:
	img = tf.io.read_file( img_file )
	img = tf.image.decode_jpeg(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)

	#for model 1
	img1 = tf.image.resize(img, [IMG_WIDTH_1, IMG_HEIGHT_1])
	img1 = np.expand_dims(img1, axis=0)
	img1 = img1.reshape(1, IMG_WIDTH_1, IMG_HEIGHT_1, 3)

	prediction1 = nn_model_1.predict_classes(img1)
	#--------------------------------------------------------
	
	#for model 2
	img2 = tf.image.resize(img, [IMG_WIDTH_2, IMG_HEIGHT_2])
	img2 = np.expand_dims(img2, axis=0)
	img2 = img2.reshape(1, IMG_WIDTH_2, IMG_HEIGHT_2, 3)

	prediction2 = nn_model_2.predict_classes(img2)
	#--------------------------------------------------------

		
	img_path = img_file.numpy()
	dest_path = os.path.split(img_path)
	parent_dir = os.path.split(dest_path[0])
	
	model_1_result = 1
	model_2_result = 1
	
	#is model_1 correct?
	if parent_dir[1].decode() != CLASS_NAMES[ prediction1[0] ]:			
		model_1_result = 0 #model_1 wrong
	
	#is model_2 correct?
	if parent_dir[1].decode() != CLASS_NAMES[ prediction2[0] ]:			
		model_2_result = 0 #model_2 wrong
	
	#add results to list
	results.append( (model_1_result, model_2_result) )
	print( (model_1_result, model_2_result) )
	if model_1_result == 1 and model_2_result == 1:
		yes_yes = yes_yes + 1
	elif model_1_result == 1 and model_2_result == 0:
		yes_no = yes_no + 1
	elif model_1_result == 0 and model_2_result == 1:
		no_yes = no_yes + 1
	else:
		no_no = no_no + 1
	print("------------------------------------------------------")

n1 = 0
n2 = 0
for tup in results:
	n1 = n1 + tup[0]
	n2 = n2 + tup[1]
print("Accurary model 1: " + str(n1/len(results)) )
print("Accurary model 2: " + str(n2/len(results)) )

#create table
table = [[yes_yes, yes_no],
		 [no_yes , no_no]]

print(table)

# calculate mcnemar test
result = mcnemar(table, exact=True) #binomial dist

# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')



input("Fin")
