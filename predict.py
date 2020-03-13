#program that uses a trained model to classify a image


#imports
#-----------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import argparse


#command-line parameters
#-----------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--img_file', type=str,required=True,
	help='Image file to classify containing the test set'
)

parser.add_argument('--model_file', type=str,required=True,
	help='HDF5 file'
)

parser.add_argument('--labels_file', type=str,required=True,
	help='File containing the string representation of a label list'
)

parser.add_argument('--arch', type=str,required=True,
	help='Neural network architecture: <resnet|googlenet>'
)


FLAGS, unparsed = parser.parse_known_args()
	

#param validation
#-----------------------------------------------------------------------
if FLAGS.arch not in ('resnet', 'googlenet'):
	print("Error: 'arch' must be 'resnet' or 'googlenet'", 
			file=sys.stderr)
	quit()



#-----------------------------------------------------------------------
if FLAGS.arch == 'resnet':
	IMG_WIDTH 	= 224
	IMG_HEIGHT	= 224
else:
	IMG_WIDTH 	= 299
	IMG_HEIGHT	= 299

label_file = open(FLAGS.labels_file)
CLASS_NAMES = []
file_text = label_file.readline()
labels = file_text[1:-1] #remove square brackets
labels = labels.split(',')
for l in labels:
	l = l.strip() # remove spaces at the beginning and ending
	CLASS_NAMES.append( l[1:-1] ) # remove single quotes

nn_model = tf.keras.models.load_model( FLAGS.model_file)


img = tf.io.read_file( FLAGS.img_file )
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
img = np.expand_dims(img, axis=0)
img = img.reshape(1, IMG_WIDTH, IMG_WIDTH, 3)

prediction = nn_model.predict(img)
print( "Model prediction: " + CLASS_NAMES[ np.argmax(prediction[0]) ] )
print("------------")

print("Probabilities:")
i = 0
for p in prediction[0]:
	print( str(CLASS_NAMES[i])+ ":\t" + str(round(p*100, 7)) + "%")
	i = i + 1
