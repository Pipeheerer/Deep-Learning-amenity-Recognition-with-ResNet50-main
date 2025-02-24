# evaluate.py

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# Load the best saved model
model = load_model('best_model.h5')

# Paths to your test data
test_data_dir = 'data/test'

# Image dimensions
img_width, img_height = 224, 224

# Data generator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

# Evaluate the model
scores = model.evaluate(test_generator)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])
