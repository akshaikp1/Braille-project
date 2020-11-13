import numpy as np
from tensorflow import keras
import tensorflow.python.keras.models
from tensorflow.python.keras.models import model_from_json, load_model
from scipy.misc.pilutil import imread, imresize,imshow
import tensorflow as tf

graph = tf.compat.v1.get_default_graph()
def init():
	with graph.as_default():
		loaded_model = load_model('model_new.hdf5')
		print("Loaded Model from disk")

		#compile and evaluate loaded model
		loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		return loaded_model,graph
