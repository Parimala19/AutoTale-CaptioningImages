import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model

# Page Config
st.set_page_config(page_title="AutoTale - AI Caption Generator", layout="centered")

#  Blue Themed UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4e8ef7;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📸 AutoTale: AI-Based Image Caption Generator")
st.write("Upload an image and get a creative caption suitable for LinkedIn, Instagram, or YouTube Shorts!")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max_length
with open("max_length.pkl", "rb") as f:
    max_length = pickle.load(f)

# Load trained model
model = tf.keras.models.load_model("caption_model.h5")

# InceptionV3 Feature Extractor (without top)
def get_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model

feature_model = get_feature_extractor()

# Preprocess image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    return feature

# Generate caption
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.replace('startseq', '').replace('endseq', '').strip().capitalize()
    return final

# File uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Save temporarily
    img_path = os.path.join("temp_uploaded.jpg")
    with open(img_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Show uploaded image
    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    # Extract features and generate caption
    with st.spinner("Generating caption..."):
        features = extract_features(img_path, feature_model)
        caption = generate_caption(model, tokenizer, features, max_length)

    # Display result
    st.success("Your Caption:")
    st.markdown(f"<h3 style='color:#4e8ef7'>{caption}</h3>", unsafe_allow_html=True)

    # Suitable for sharing
    st.write("Copy and use this caption for your post on:")
    st.markdown("- LinkedIn")
    st.markdown("- Instagram")
    st.markdown("- YouTube Shorts")
