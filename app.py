from flask import Flask, request
import cv2
import numpy as np
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from pygame import mixer
import threading

app = Flask(__name__)

mixer.init()
mixer.music.load("music.mp3")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

thresh = 0.25
flag = 0
frame_check = 20

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def process_frame(frame):
    global flag
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)
        ear = (leftEar + rightEar) / 2.0

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                print("ALERT: Drowsiness detected!")
                mixer.music.play()
        else:
            flag = 0
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        file_bytes = request.data
        np_frame = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

        threading.Thread(target=process_frame, args=(frame,)).start()
    except Exception as e:
        print(f"Error processing frame: {e}")
    return '', 204 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
