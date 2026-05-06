# Ghana Sign Language (GSL) Hand Gesture Recognition

## 📌 Overview

This project is a real-time hand gesture recognition system trained on Ghana Sign Language (GSL).
It uses computer vision and machine learning to detect and classify hand gestures through a webcam.

---

## 🚀 Features

* Real-time hand tracking using MediaPipe
* Custom dataset collection system
* Machine learning model for gesture classification
* Live gesture prediction with confidence score

---

## 🛠️ Technologies Used

* Python
* OpenCV
* MediaPipe
* Scikit-learn
* NumPy

---

## 🧠 System Pipeline

Webcam → MediaPipe → Landmark Extraction → ML Model → Prediction

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run hand detection

```bash
python test_hand_detection.py
```

### 3. Collect data

```bash
python collect_data.py
```

### 4. Train model

```bash
python train_model.py
```

### 5. Run real-time prediction

```bash
python real_time_predict.py
```

---

## 📊 Dataset

The dataset consists of hand landmark coordinates captured using MediaPipe for different GSL gestures.

---

## 🎯 Future Improvements

* Add more gestures
* Improve model accuracy
* Convert to full sentence recognition
* Deploy as a web application

---

## 👨‍💻 Author

Kojo Baafi Botwe
