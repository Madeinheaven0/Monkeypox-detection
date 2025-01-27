import streamlit as st
from tensorflow import keras 
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

#Load the model
model = keras.models.load_model("monkeypox_detection.keras")
classes = ["positif", "negatif"]

st.title("MONKEYPOX DETECTION")

#Download the image
image_file = st.file_uploader("Select an image", type=["png", "jpg", "jepg"])

if image_file:
    
    #Load the image
    image = Image.open(image_file)
    st.image(image, caption="Image of the patient", use_container_width=True)
    
    #Resize the image to the dimensions of a xception
    image = image.resize((299, 299))
    
    #Convert the image to an array
    image = img_to_array(image)

    #Add an supplementary dimensions to match at the format waited (batch size)
    image = np.expand_dims(image, axis=0)

    #Preprocess the image with the process_input of xception
    preprocessed_image = preprocess_input(image)

    predictions = model.predict(preprocessed_image)

    if classes[np.argmax(predictions)] == classes[0]:
        st.text(f"You have monkeypox or most especially you have {predictions[0, 0] * 100:.5f}% risk to have monkeypox.")
    else:
        st.text(f"You dn't have monkeypox or most especially you have {predictions[0, 1] * 100:.5f}% to don't have monkeypox. Your problem have another origins")