import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import os
from tqdm import tqdm
import cv2 as cv


model = load_model('C:/Users/Admin/PycharmProjects/SentimentAnalysisCV/senti.h5')


haar = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv.CascadeClassifier(haar)


def extract_features(image):
    image = cv.resize(image, (48, 48))
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0



labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        face_image = gray[q:q + s, p:p + r]
        cv.rectangle(frame, (p, q), (p + r, q + s), (0, 255, 0), 2)

        try:
            img = extract_features(face_image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv.putText(frame, '%s' % prediction_label, (p, q - 10), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        except Exception as e:
            print(f"Prediction error: {e}")

    cv.imshow('Output', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
