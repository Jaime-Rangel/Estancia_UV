
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))