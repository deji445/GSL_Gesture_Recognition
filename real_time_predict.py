import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

model = joblib.load("gsl_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

sentence = ""
last_prediction = ""
last_added_time = 0

confidence_threshold = 80
delay_seconds = 1.2

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    display_text = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            row = []

            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row = np.array(row).reshape(1, -1)

            prediction = model.predict(row)
            gesture = label_encoder.inverse_transform(prediction)[0]

            if hasattr(model, "predict_proba"):
                confidence = max(model.predict_proba(row)[0]) * 100
            else:
                confidence = 0

            display_gesture = gesture.upper()

            if display_gesture == "SPACE":
                display_text = "[SPACE]"
            elif display_gesture == "DELETE":
                display_text = "[DELETE]"
            else:
                display_text = gesture

            current_time = time.time()

            if (
                confidence >= confidence_threshold
                and gesture != last_prediction
                and current_time - last_added_time >= delay_seconds
            ):
                if display_gesture == "SPACE":
                    sentence += " "
                elif display_gesture == "DELETE":
                    sentence = sentence[:-1]
                else:
                    sentence += gesture

                last_prediction = gesture
                last_added_time = current_time

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            cv2.putText(
                frame,
                f"{display_text} ({confidence:.1f}%)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"Text: {sentence}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press C to clear | Press Q to quit",
        (10, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("GSL Real-Time Prediction", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        sentence = ""
        last_prediction = ""

cap.release()
cv2.destroyAllWindows()
