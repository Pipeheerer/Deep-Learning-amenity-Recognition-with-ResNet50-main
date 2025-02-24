import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Streamlit app
st.image('logo.png')
st.markdown("## Amenity Identifier App with Deep Learning")
st.markdown("""
This app uses Deep learning ResNet50 libraries, a convolutional neural network that is 50 layers deep, namely keras to identify objects from images.
Disclaimer: This Prototype is for demonstration and educational purposes only. It is not intended for production use and may not be reliable or accurate in all situations.

**Made by Dennis Marisa**
""")

# Uploading multiple images
object_images = st.file_uploader("Upload images...", type=['png', 'jpg', 'webp', 'jpeg'], accept_multiple_files=True)
submit = st.button('Predict')

if submit:
    if object_images is not None:
        all_predictions = []
        
        for object_image in object_images:
            # Convert the file to an opencv image
            file_bytes = np.asarray(bytearray(object_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            
            # Display the uploaded image
            st.image(opencv_image, channels="BGR", caption="Uploaded Image", use_column_width=True)
            
            # Preprocess the image
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            opencv_image = cv2.resize(opencv_image, (224, 224))
            opencv_image = np.expand_dims(opencv_image, axis=0)
            opencv_image = preprocess_input(opencv_image)
            
            # Make predictions
            predictions = model.predict(opencv_image)
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            
            # Collect all predictions
            all_predictions.extend(decoded_predictions)
        
        # Sort predictions by likelihood and get top 10
        all_predictions.sort(key=lambda x: x[2], reverse=True)
        top_10_predictions = all_predictions[:10]
        
        # Display top 10 predictions
        st.markdown("### Top Amenity Predictions Across All Images:")
        for imagenet_id, name, likelihood in top_10_predictions:
            st.text(f'- {name}: {likelihood:.2f} likelihood')