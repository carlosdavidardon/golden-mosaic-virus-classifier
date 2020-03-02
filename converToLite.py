#program to convert a standar tensorflow model
#into a tensorflow lite model


#imports
#-----------------------------------------------------------------------
import tensorflow as tf
import argparse

#command-line parameters
#-----------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--tf_file', type=str,required=True,
	default=None,
	help='Directory to load the trained model as a HDF5 file'
)

parser.add_argument('--tflite_file', type=str,required=True,
	default=None,
	help='Directory to save the trained model as a Tensorflow lite file'
)

parser.add_argument('--tflite_type', type=str,required=False,
	default='float',
	help="'float' or 'quantized' (smaller)"
)

FLAGS, unparsed = parser.parse_known_args()
#-----------------------------------------------------------------------

#load model
tf_model = tf.keras.models.load_model( FLAGS.tf_file )
tf_model.summary()

#convert and save lite model
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
if FLAGS.tflite_type == 'quantized':
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	print("Quantized model.")
tflite_model = converter.convert()
open(FLAGS.tflite_file, "wb").write(tflite_model)

print("Fin")

