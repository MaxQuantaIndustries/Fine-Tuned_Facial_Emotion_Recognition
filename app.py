import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "train2/emotion_mobilenetv2_finetuned.keras"  # saved model path
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Angry", "Happy", "Sad"]  # must match your training order

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_emotion_model():
    model = load_model(MODEL_PATH)
    return model

model = load_emotion_model()

# -------------------------
# PREDICT FUNCTION
# -------------------------
def predict_emotion(face_rgb):
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    face_normalized = face_resized.astype("float32") / 255.0
    face_batch = np.expand_dims(face_normalized, axis=0)
    preds = model.predict(face_batch, verbose=0)
    idx = np.argmax(preds[0])
    return CLASS_NAMES[idx], float(preds[0][idx]), preds[0]

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="centered")

# Headline Section
st.markdown(
    "<h2 style='text-align: center;'>Emotion → Angry, Happy, Sad</h2>",
    unsafe_allow_html=True
)

st.title("Image-Based Emotion Detection")
st.write("Upload an image and let the model detect emotions.")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Convert to OpenCV format
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        st.warning("No face detected in the image.")
        st.image(img_array, caption="Uploaded Image", use_column_width=True)
    else:
        for (x, y, w, h) in faces:
            face_rgb = img_array[y:y+h, x:x+w]
            label, conf, probs = predict_emotion(face_rgb)

            # Draw rectangle & label
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{label} ({conf*100:.1f}%)"
            cv2.putText(img_array, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

        st.image(img_array, caption="Detected Face(s) & Prediction", use_column_width=True)

        # -------------------------
        # ANALYTICS GRAPH
        # -------------------------
        st.subheader("Emotion Probability Distribution")

        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, probs, color=["red", "green", "blue"])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")
        ax.set_title("Model Prediction Confidence")

        for i, v in enumerate(probs):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

        st.pyplot(fig)

        # Metrics
        st.metric(label="Predicted Emotion", value=label, delta=f"{conf*100:.1f}% confidence")
