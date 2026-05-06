import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Ghana Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)
st.markdown(
    """
    <h1 style='text-align:center; color:#00E5FF;'>
     Ghana Sign Language Recognition
    </h1>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = joblib.load("gsl_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

@st.cache_resource
def load_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    return mp_hands, hands

model, label_encoder = load_model()
mp_hands, hands = load_hands()
mp_draw = mp.solutions.drawing_utils

# ---------------- SIDEBAR ----------------

st.sidebar.markdown(
    """
    <h1 style='color:#00E5FF;'>📌 Instructions</h1>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("""
### 🖐 How To Use
1. Show your hand gesture clearly  
2. Ensure good lighting  
3. Keep one hand visible  
4. Capture image using webcam  
5. Wait for prediction  

---

### ✨ Special Gestures

🖐 **SPACE** → Open Palm  

👍 **DELETE** → Thumbs Down  

---

### ✅ Supported Features
- GSL Alphabet Recognition  
- Space Gesture  
- Delete Gesture  
- AI-Based Prediction  
- Confidence Scoring  

---

### 🛠 Tech Stack
- Python
- Streamlit
- MediaPipe
- OpenCV
- Scikit-learn
""")

st.divider()

# ---------------- LAYOUT ----------------

col1, col2 = st.columns([1.2, 1])

# ---------------- CAMERA SECTION ----------------

with col1:
    st.subheader("📷 Capture Gesture")

    camera_image = st.camera_input("Take a picture")

# ---------------- PREDICTION SECTION ----------------

with col2:
    st.subheader("🧠 Prediction Results")

    if camera_image is not None:

        image = Image.open(camera_image)
        frame = np.array(image)

        result = hands.process(frame)

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                row = []

                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                row = np.array(row).reshape(1, -1)

                prediction = model.predict(row)
                gesture = label_encoder.inverse_transform(prediction)[0]

                confidence = max(model.predict_proba(row)[0]) * 100

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Prediction Card
                st.success(f"### ✨ Prediction: {gesture}")

                # Confidence Bar
                st.progress(int(confidence))

                st.info(f"Confidence: {confidence:.1f}%")

                # Confidence Warning
                if confidence < 70:
                    st.warning(
                        "Low confidence prediction. "
                        "Try improving lighting or hand positioning."
                    )

            # Show processed image
            st.image(
                frame,
                caption="Detected Hand Landmarks",
                use_container_width=True
            )

        else:
            st.error("❌ No hand detected. Try again.")

# ---------------- GESTURE GUIDE ----------------

st.divider()

st.subheader("📖 Ghana Sign Language Gesture Guide")

st.image(
    "gesture_chart.jpg",
    caption="Supported Ghana Sign Language Alphabet Gestures",
     width=500
)

# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    """
    <center>
    Developed using Machine Learning and Computer Vision  
    for Ghana Sign Language Accessibility
    </center>
    """,
    unsafe_allow_html=True
)