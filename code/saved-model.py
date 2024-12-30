# Save a model in a special format called tensorflow `SavedModel`.

import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('./geo-model.keras')

# Export the model in SavedModel format
model.export('saved-geo-model')

print("Model exported successfully in SavedModel format at: saved-geo-model")
