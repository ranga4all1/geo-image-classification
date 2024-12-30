# Save a model in a special format called tensorflow `SavedModel`.

import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('./geo-model.keras')

tf.saved_model.save(model, 'geo-model')
