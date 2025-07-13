import os
import gdown
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AutoTale - AI Image Captioning", page_icon="📸", layout="centered")

# ---------------- UI HEADER ----------------
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtext {
        text-align: center;
        font-size: 18px;
        color: #34495E;
        margin-bottom: 40px;
    }
    </style>
    <div class="title">📸 AutoTale: AI Image Caption Generator</div>
    <div class="subtext">Upload an image and let AI generate a caption perfect for Instagram, LinkedIn, or YouTube Shorts!</div>
""", unsafe_allow_html=True)

# ---------------- Download Model If Not Exists ----------------
MODEL_PATH = "caption_model.h5"
MODEL_DRIVE_ID = "1O3p-ZHHn9ANhVEGLbR4ST5rHcxhxMEjd"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"

@st.cache_resource
def load_caption_model():
    if not os.path.exists(MODEL_PATH):
        st.info(" Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_caption_model()

# ---------------- Load Tokenizer & Feature Extractor ----------------
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

max_length = 34  # Set based on your preprocessing

# ---------------- Generate Caption ----------------
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([photo, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = tokenizer.index_word.get(y_pred, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip().capitalize()

# ---------------- Image Upload & Processing ----------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        st.info("Extracting image features...")
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        from tensorflow.keras.models import Model
        from tensorflow.keras.preprocessing.image import img_to_array

        # Feature extractor
        def extract_features(image):
            model = InceptionV3(weights='imagenet')
            model = Model(inputs=model.input, outputs=model.layers[-2].output)
            image = image.resize((299, 299))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            return model.predict(image)

        photo_features = extract_features(image)

        st.success("Features extracted. Generating caption...")
        caption = generate_caption(model, tokenizer, photo_features, max_length)
        
        st.markdown(f"""
        <div style="background-color: #ecf0f1; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #2980b9;"> Generated Caption:</h4>
            <p style="font-size: 20px; color: #2c3e50;"><em>"{caption}"</em></p>
        </div>
        """, unsafe_allow_html=True)
