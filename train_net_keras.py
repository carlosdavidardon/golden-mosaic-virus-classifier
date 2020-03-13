#program to transfer learning from resnet or googlenet into a
#specific problem with a custom dataset


#imports
#-----------------------------------------------------------------------
import tensorflow as tf
import argparse
import numpy as np
import pathlib
import sys
import matplotlib.pyplot as plt
import os

from tensorflow.python.platform import gfile




#command-line parameters
#-----------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', type=str,required=True,
	help='Folder containing the training set'
)

parser.add_argument('--val_dir', type=str,required=True,
	help='Folder containing the validation set'
)

parser.add_argument('--test_dir', type=str,required=True,
	help='Folder containing the test set'
)

parser.add_argument('--arch', type=str,required=True,
	help='Neural network architecture: <resnet|googlenet|mobilenet>'
)

parser.add_argument('--epochs', type=int,required=False,
	default=15,
	help='Number of epochs to train the neural network'
)

parser.add_argument('--batch_size', type=int,required=False,
	default=32,
	help='Number of images per batch'
)

parser.add_argument('--optimizer', type=str,required=False,
	default='rmsprop',
	help='Keras optimizer: <adam|rmsprop>'
)

parser.add_argument('--learning_rate', type=float,required=False,
	default=0.001,
	help='Float value >= 0'
)

parser.add_argument('--dropout', type=float,required=False,
	default=0.0,
	help='Float between 0 and 1. Fraction of the input units to drop'
)

parser.add_argument('--save_checkpoints_dir', type=str,required=False,
	default=None,
	help='Directory to save checkpoints'
)

parser.add_argument('--load_checkpoints_dir', type=str,required=False,
	default=None,
	help='Directory containing previous saved checkpoints'
)

parser.add_argument('--trained_model_file', type=str,required=False,
	default=None,
	help='Directory to save the trained model as an HDF5 file'
)

parser.add_argument('--log_dir', type=str,required=False,
	default=None,
	help='Directory to save text logs'
)


FLAGS, unparsed = parser.parse_known_args()



#param validation
#-----------------------------------------------------------------------
if FLAGS.arch not in ('resnet', 'googlenet', 'mobilenet'):
	print("Error: 'arch' must be 'resnet', 'googlenet' or 'mobilenet'", 
			file=sys.stderr)
	quit()

if FLAGS.epochs <= 0:
	print("Error: 'epochs' must be greater than 0", 
			file=sys.stderr)
	quit()

if FLAGS.batch_size <= 0:
	print("Error: 'batch_size' must be greater than 0", 
			file=sys.stderr)
	quit()

if FLAGS.optimizer not in ('adam', 'rmsprop'):
	print("Error: 'optimizer' must be 'adam' or 'rmsprop'", 
			file=sys.stderr)
	quit()

if FLAGS.learning_rate < 0:
	print("Error: 'learning_rate' must be greater than 0", 
			file=sys.stderr)
	quit()

if FLAGS.dropout < 0 and FLAGS.dropout >= 1:
	print("Error: 'dropout' must be between 0 and 1", 
			file=sys.stderr)
	quit()

if FLAGS.save_checkpoints_dir != None and FLAGS.save_checkpoints_dir == FLAGS.load_checkpoints_dir:
	print("Error: Directory to save and load checkpoints cannot be the same", 
			file=sys.stderr)
	quit()
	




#global variables
#-----------------------------------------------------------------------
CLASS_NAMES 		= None

BATCH_SIZE 			= FLAGS.batch_size
SHUFFLE_BUFFER_SIZE = 1000
EPOCHS 				= FLAGS.epochs
IMG_WIDTH 			= None
IMG_HEIGHT			= None
IMAGE_COUNT 		= 0
STEPS_PER_EPOCH 	= 0

VAL_IMG_COUNT 		= 0
VAL_STEPS_PER_EPOCH = 0

TEST_IMG_COUNT		= 0
TEST_STEPS			= 0


AUTOTUNE 			= tf.data.experimental.AUTOTUNE


#-----------------------------------------------------------------------
def load_sample_metadata(type='train'):
	global CLASS_NAMES
	global IMAGE_COUNT
	global VAL_IMG_COUNT
	global TEST_IMG_COUNT
	global STEPS_PER_EPOCH
	global VAL_STEPS_PER_EPOCH
	global TEST_STEPS

	if type == 'train':
		imgset = FLAGS.train_dir
	elif type == 'val':
		imgset = FLAGS.val_dir
	else:
		imgset = FLAGS.test_dir
		
	
	print("Loading sample metadata...")
	
	if not gfile.Exists(imgset):
		print("Image directory '" + imgset + "' not found.", 
				file=sys.stderr)
		quit()

	data_dir = pathlib.Path(imgset)	
	if type == 'train':
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

	elif type == 'val':
		#TODO: class names in val must be equal to train
		VAL_IMG_COUNT = len(list(data_dir.glob('*/*.jpg') )
							+list(data_dir.glob('*/*.JPG'))
							+list(data_dir.glob('*/*.jpeg'))
							+list(data_dir.glob('*/*.JPEG')) )

		VAL_STEPS_PER_EPOCH = np.ceil( VAL_IMG_COUNT/BATCH_SIZE )
		print("Validation steps per epoch: "+str(VAL_STEPS_PER_EPOCH))
	else:
		#TODO: class names in val must be equal to train
		TEST_IMG_COUNT = len(list(data_dir.glob('*/*.jpg') )
							+list(data_dir.glob('*/*.JPG'))
							+list(data_dir.glob('*/*.jpeg'))
							+list(data_dir.glob('*/*.JPEG')) )

		TEST_STEPS = np.ceil( TEST_IMG_COUNT /BATCH_SIZE )
		print("Test steps: "+str(TEST_STEPS))

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
def create_model():
	if FLAGS.arch == 'resnet':
		base_model = tf.keras.applications.ResNet50V2(
											input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
											include_top=False,
											weights='imagenet')
	elif FLAGS.arch == 'mobilenet':
		base_model = tf.keras.applications.MobileNetV2(
											input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
											include_top=False,
											weights='imagenet')
	else:
		base_model = tf.keras.applications.InceptionV3(
											input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
											include_top=False,
											weights='imagenet')
		
	for layer in base_model.layers:
		layer.trainable = False

	x = base_model.output
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	
	if FLAGS.dropout > 0.0:
		x = tf.keras.layers.Dropout(FLAGS.dropout)(x)
	
	predictions = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
	
	model = tf.keras.models.Model(inputs=base_model.inputs, outputs=predictions)
	
	model.compile(loss='categorical_crossentropy', 
				  optimizer=tf.keras.optimizers.RMSprop(lr=FLAGS.learning_rate),
				  metrics=['accuracy'])
	return model
	
	
#-----------------------------------------------------------------------
if FLAGS.arch == 'resnet' or FLAGS.arch == 'mobilenet':
	IMG_WIDTH 	= 224
	IMG_HEIGHT	= 224
else:
	IMG_WIDTH 	= 299
	IMG_HEIGHT	= 299


load_sample_metadata('train')
load_sample_metadata('val')
load_sample_metadata('test')



nn_model = create_model()
print("Neural network model created.")
nn_model.summary()


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
																shear_range=0.2,
																zoom_range=0.2,
																rotation_range=90,
																width_shift_range=0.1,
																height_shift_range=0.1,
																brightness_range=(0.6, 1.5),
																horizontal_flip=True,
																vertical_flip=True
																)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

print("Target size: "+str(IMG_WIDTH)+"x"+str(IMG_HEIGHT))

train_dataset = train_datagen.flow_from_directory(FLAGS.train_dir, class_mode='categorical', 
										batch_size=BATCH_SIZE,
										target_size=(IMG_WIDTH, IMG_HEIGHT))
val_dataset = val_datagen.flow_from_directory(FLAGS.val_dir, class_mode='categorical', 
										batch_size=BATCH_SIZE,
										target_size=(IMG_WIDTH, IMG_HEIGHT))
test_dataset = test_datagen.flow_from_directory(FLAGS.test_dir, class_mode='categorical', 
										batch_size=BATCH_SIZE,
										target_size=(IMG_WIDTH, IMG_HEIGHT))


print("Train dataset size: "+str(IMAGE_COUNT))
print("Validation dataset size: "+str(VAL_IMG_COUNT))
print("Test dataset size: "+str(TEST_IMG_COUNT))



													
#Loads the weights (resume training)
if FLAGS.load_checkpoints_dir != None:
	latest = tf.train.latest_checkpoint(FLAGS.load_checkpoints_dir)
	nn_model.load_weights(latest)
	print("Latest weights loaded from checkpoints")


callback_list = None
# Save the weights using the `checkpoint_path` format
if FLAGS.save_checkpoints_dir != None:
	checkpoint_path 	= os.path.join(FLAGS.save_checkpoints_dir, 
										"cp-{epoch:04d}.ckpt")
	
	#callback that saves the model's weights
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
													save_weights_only=True,
													verbose=1,
													period=1)
	nn_model.save_weights(checkpoint_path.format(epoch=0))
	callback_list = [cp_callback]

history = nn_model.fit(train_dataset,
						epochs=EPOCHS,
						steps_per_epoch=STEPS_PER_EPOCH,
						validation_data=val_dataset,
						validation_steps=VAL_STEPS_PER_EPOCH,
						callbacks=callback_list)


# Fine tunning
print("Fine-tunning")

if FLAGS.arch == 'resnet':
	for layer in nn_model.layers[:-15]:
		layer.trainable = False
			
	for layer in nn_model.layers[-15:]:
		layer.trainable = True
elif FLAGS.arch == 'googlenet':
	for layer in nn_model.layers[:-33]:
		layer.trainable = False
			
	for layer in nn_model.layers[-33:]:
		layer.trainable = True


nn_model.compile(optimizer=tf.keras.optimizers.SGD(lr=FLAGS.learning_rate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

nn_model.summary()

history2 = nn_model.fit(train_dataset,
						epochs=5,
						steps_per_epoch=STEPS_PER_EPOCH,
						validation_data=val_dataset,
						validation_steps=VAL_STEPS_PER_EPOCH)






# Save the entire model to a HDF5 file.
h5_file = None
if FLAGS.trained_model_file == None:
	h5_file = FLAGS.arch+'.h5'
else:
	h5_file = FLAGS.trained_model_file

print("Saving model: "+h5_file)
nn_model.save(h5_file)


#report test accuracy
print("Test accuracy:")
nn_model.evaluate(test_dataset, steps=TEST_STEPS)


acc = history.history['accuracy'] + history2.history['accuracy']
val_acc = history.history['val_accuracy'] + history2.history['val_accuracy']

loss = history.history['loss'] + history2.history['loss']
val_loss = history.history['val_loss'] + history2.history['val_loss']


#save log files
if FLAGS.log_dir != None:
	label_file = open( os.path.join(FLAGS.log_dir, "labels.txt"), "w")
else:
	label_file = open("labels.txt", "w")
label_file.writelines(str(CLASS_NAMES.tolist()))
label_file.close()

if FLAGS.log_dir != None:
	results_file = open( os.path.join(FLAGS.log_dir, "results.txt"), "w")
else:
	results_file = open("results.txt", "w")
results_file.writelines(str(acc)+"\n")
results_file.writelines(str(val_acc)+"\n")
results_file.writelines(str(loss)+"\n")
results_file.writelines(str(val_loss))
results_file.close()



#plot graph
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Categorical Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


input("Fin")
