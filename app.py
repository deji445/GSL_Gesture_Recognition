import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import time

st.set_page_config(
    page_title="Ghana Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = joblib.load("gsl_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

model, label_encoder = load_model()

# ---------------- UI ----------------

st.sidebar.markdown("<h1 style='color:#00E5FF;'>📌 Instructions</h1>", unsafe_allow_html=True)

st.sidebar.markdown("""
### 🖐 How To Use
1. Click **START**
2. Allow camera access
3. Show one hand clearly
4. Keep good lighting

---

### ✨ Special Gestures
🖐 **SPACE** → Open Palm  
👍 **DELETE** → Thumbs Down  

---

### ✅ Supported Features
- Live Webcam Detection
- GSL Alphabet Recognition
- Confidence Scoring
- MediaPipe Hand Tracking
""")

st.markdown(
    """
    <h1 style='text-align:center; color:#00E5FF;'>
    🤟 Ghana Sign Language Recognition
    </h1>
    <h4 style='text-align:center; color:#BBBBBB;'>
    Live AI-powered Ghana Sign Language recognition
    </h4>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- VIDEO PROCESSOR ----------------

class GSLVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.sentence = ""
        self.last_prediction = ""
        self.last_added_time = 0

        self.confidence_threshold = 80
        self.delay_seconds = 1.2

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                row = []

                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                row = np.array(row).reshape(1, -1)

                prediction = model.predict(row)
                gesture = label_encoder.inverse_transform(prediction)[0]

                confidence = max(model.predict_proba(row)[0]) * 100

                display_gesture = gesture.upper()

                if display_gesture == "SPACE":
                    display_text = "[SPACE]"
                elif display_gesture == "DELETE":
                    display_text = "[DELETE]"
                else:
                    display_text = gesture

                current_time = time.time()

                if (
                    confidence >= self.confidence_threshold
                    and gesture != self.last_prediction
                    and current_time - self.last_added_time >= self.delay_seconds
                ):

                    if display_gesture == "SPACE":
                        self.sentence += " "

                    elif display_gesture == "DELETE":
                        self.sentence = self.sentence[:-1]

                    else:
                        self.sentence += gesture

                    self.last_prediction = gesture
                    self.last_added_time = current_time

                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                cv2.putText(
                    img,
                    f"{display_text} ({confidence:.1f}%)",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 255, 0),
                    3
                )

        # SENTENCE DISPLAY
        cv2.rectangle(img, (10, 80), (1200, 140), (0, 0, 0), -1)

        cv2.putText(
            img,
            f"Text: {self.sentence}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- LIVE STREAM ----------------

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.subheader("🎥 Live Webcam Detection")

webrtc_streamer(
    key="gsl-live-detection",
    video_processor_factory=GSLVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)