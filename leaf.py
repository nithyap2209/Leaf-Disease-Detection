import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

# Load the trained model
model = tf.keras.models.load_model('leaf_disease_model.h5')

# Load class names from the JSON file
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Define image preprocessing function
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize image to match model input size
    image_array = np.array(image) / 255.0  # Normalize image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Define Streamlit app
st.title("Leaf Disease Detection")
st.write("Upload an image of a leaf to classify its disease.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_array = preprocess_image(image)

    # Predict the class of the image
    predictions = model.predict(image_array)
    class_index = np.argmax(predictions[0])

    # Check if class_index is within the range of available class names
    if class_index < len(class_names):
        class_name = class_names[class_index]
        # Display the result
        st.write(f"Prediction: {class_name}")
        st.write(f"Confidence: {predictions[0][class_index] * 100:.2f}%")
    else:
        st.write("Error: The model's class index is out of range.")
