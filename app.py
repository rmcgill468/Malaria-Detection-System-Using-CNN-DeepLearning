import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🔬",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    }
    .title {
        text-align: center;
        color: #00d4ff;
        font-size: 80px;
        font-weight: bold;
        padding: 20px;
        text-shadow: 2px 2px 10px #00d4ff;
    }
    .subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .infected {
        background-color: #ff4b4b;
        color: white;
    }
    .healthy {
        background-color: #00c853;
        color: white;
    }
    .info-box {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: #1a1a2e;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 30px;
        width: 100%;
        font-size: 18px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #1a1a2e;
    }
    </style>
""", unsafe_allow_html=True)

model = load_model("malaria_model.h5")
IMG_SIZE = 64

st.markdown('<h1 style="text-align:center; color:#00d4ff; font-size:50px; font-weight:bold; text-shadow:2px 2px 10px #00d4ff;">🔬 Malaria Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#ffffff; font-size:22px; margin-bottom:30px;">Upload a blood cell image to detect malaria infection using AI</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class="info-box">
            <h4 style="color:#00d4ff;">🧠 AI Powered</h4>
            <p style="font-size:20px;">Deep CNN model trained on 27,558 cell images</p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="info-box">
            <h4 style="color:#00d4ff;">⚡ Fast Results</h4>
            <p style="font-size:20px;">Get instant predictions in seconds</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class="info-box">
            <h4 style="color:#00d4ff;">🎯 High Accuracy</h4>
            <p style="font-size:20px;">Trained with advanced data augmentation</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload Blood Cell Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("""
            <div class="info-box">
                <h4 style="color:#00d4ff;">📊 Image Details</h4>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<p style="color:white; font-size:15px;"><b>Filename:</b> {uploaded_file.name}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:white; font-size:15px;"><b>Size:</b> {image.size}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:white; font-size:15px;"><b>Format:</b> {image.format}</p>', unsafe_allow_html=True)

    if st.button("🔍 Analyze Cell"):

        with st.spinner("Analyzing cell image..."):

            img = np.array(image)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.reshape(img, [1, IMG_SIZE, IMG_SIZE, 3])

            prediction = model.predict(img)
            confidence = prediction[0][0]

        st.markdown("---")
        st.markdown('<h3 style="color:#00d4ff;">📋 Analysis Result</h3>', unsafe_allow_html=True)

        if confidence > 0.5:
            st.markdown(f"""
                <div class="result-box healthy">
                    ✅ UNINFECTED CELL<br>
                    <span style="font-size:16px;">Confidence: {confidence*100:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box infected">
                    ⚠️ PARASITIZED CELL<br>
                    <span style="font-size:35px;">Confidence: {(1-confidence)*100:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align:center; color:#ffffff; font-size:20px;">Malaria Detection System | Built with CNN & Streamlit</p>', unsafe_allow_html=True)
