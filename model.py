import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, RMSprop

IMG_SIZE = 400
no_of_fruits = 10

def create_model():
	model = tf.keras.models.Sequential()
	model.add(tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
	model.add(tf.keras.layers.Conv2D(32,(5,5),activation='relu',padding='same'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dropout(0.25))
	model.add(tf.keras.layers.MaxPooling2D((5, 5), padding='same'))
	model.add(tf.keras.layers.Conv2D(64,(5,5),activation='relu',padding='same'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPooling2D((5, 5), padding='same'))
	model.add(tf.keras.layers.Dropout(0.25))
	model.add(tf.keras.layers.Conv2D(128,(5, 5), activation='relu',padding='same'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPooling2D((5, 5), padding='same'))
	model.add(tf.keras.layers.Conv2D(256,(5, 5), activation='relu',padding='same'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPooling2D((5, 5), padding='same'))
	model.add(tf.keras.layers.Conv2D(512,(5,5),activation='relu',padding='same'))
	model.add(tf.keras.layers.MaxPooling2D((5, 5), padding='same'))
	model.add(tf.keras.layers.Conv2D(1024,(5,5),activation='relu',padding='same'))
	model.add(tf.keras.layers.MaxPooling2D((5, 5), padding='same'))
	model.add(tf.keras.layers.GlobalAveragePooling2D())
	model.add(tf.keras.layers.Dense(256, activation='relu'))
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dropout(.2))
	model.add(tf.keras.layers.Dense(64, activation='relu'))
	model.add(tf.keras.layers.Dropout(.2))
	model.add(tf.keras.layers.Dense(32, activation='relu'))
	model.add(tf.keras.layers.Dense(no_of_fruits, activation='softmax'))
	return model
