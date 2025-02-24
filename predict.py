# predict.py

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input

# Load the best saved model
model = load_model('best_model.h5')

# Load an image file that you want to test
img_path = 'data/test/pool/img1.jpg'  # Change this path to the image you want to predict
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict
preds = model.predict(x)
predicted_class = np.argmax(preds, axis=1)

# Map predicted class index to class label
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}
print('Predicted:', class_labels[predicted_class[0]])
