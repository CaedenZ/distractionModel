# imports
import cv2
import numpy as np
import dlib
from math import hypot
from keras.models import load_model
from typing import Optional

# WORK NOTES :
# 1- 
# Error in line 199
# AttributeError: 'tuple' object has no attribute 'any'
# I have tried to make the faces to be taken from detect_face and use them for detec_emotions but I was unsuccessful

# Global values
# values for eye positions
eye_points = [36, 37, 38, 39, 40, 41]


def execution(frame, height, width):
    face, landmarks = detect_face(frame)
    if face is None:
        # Empty CI since there is no face
        return None, ""
    blinking_ratio = get_blinking_ratio(landmarks)
    lr_gaze_ratio, ud_gaze_ratio = get_gaze_ratio(landmarks, frame, height, width)
    gaze_weights = get_gaze_weights(blinking_ratio, lr_gaze_ratio)
    emotion = detect_emotion(frame)
    ci = get_ci(gaze_weights, emotion)
    return face, ci


def get_midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


# The faces in a signle frame (most of the time it's one frame)
# If there is more than one face, take the largest face (height & width)
# TODO: define a return - Mohammed
def detect_face(frame) -> Optional:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    print("len(faces) = ", len(faces))

    if len(faces) == 0:
        print("NO FACE DETECTED")
        return None, None
    # This way we ONLY consider the first face that appear.
    face = faces[0]

    predictor = dlib.shape_predictor("./util/model/shape_predictor_68_face_landmarks.dat")
    landmarks = predictor(gray, face)
    return face, landmarks


# Function for eye size
def get_blinking_ratio(facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = get_midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = get_midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    eye_ratio = ver_line_lenght / hor_line_lenght
    # We can use ratio to define object? variable eye_ratio and maybe no need to return it Idk
    # eye_ratio = ratio
    return eye_ratio

    # Gaze detection function


def get_gaze_ratio(facial_landmarks, frame, height, width):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_eye_region = np.array(
        [
            (
                facial_landmarks.part(eye_points[0]).x,
                facial_landmarks.part(eye_points[0]).y
            ),
            (
                facial_landmarks.part(eye_points[1]).x,
                facial_landmarks.part(eye_points[1]).y
            ),
            (
                facial_landmarks.part(eye_points[2]).x,
                facial_landmarks.part(eye_points[2]).y
            ),
            (
                facial_landmarks.part(eye_points[3]).x,
                facial_landmarks.part(eye_points[3]).y
            ),
            (
                facial_landmarks.part(eye_points[4]).x,
                facial_landmarks.part(eye_points[4]).y
            ),
            (
                facial_landmarks.part(eye_points[5]).x,
                facial_landmarks.part(eye_points[5]).y
            )
        ],
        np.int32
    )

    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    height, width = threshold_eye.shape

    # Calculations for Left & Right gaze ratios
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    # Calculations for Up & Down gaze ratios
    up_side_threshold = threshold_eye[0: int(height / 2), 0: int(width / 2)]
    up_side_white = cv2.countNonZero(up_side_threshold)
    down_side_threshold = threshold_eye[int(height / 2): height, 0: width]
    down_side_white = cv2.countNonZero(down_side_threshold)

    lr_gaze_ratio = (left_side_white + 10) / (right_side_white + 10)
    ud_gaze_ratio = (up_side_white + 10) / (down_side_white + 10)

    # x = lr_gaze_ratio
    # y = ud_gaze_ratio

    return lr_gaze_ratio, ud_gaze_ratio


# Calculate weights for gaze
# ud_gaze_ratio is unused in calculations
def get_gaze_weights(eye_ratio, lr_gaze_ratio):
    gaze_weights = 0

    if eye_ratio < 0.2:
        gaze_weights = 0
    elif 0.2 < eye_ratio < 0.3:
        gaze_weights = 1.5
    else:  # TODO
        if 2 > lr_gaze_ratio > 1:  # TypeError: '<' not supported between instances of 'tuple' and 'int'
            gaze_weights = 5
        else:
            gaze_weights = 2

    gaze_weights = gaze_weights
    return gaze_weights


# Helper function for detect_emotion [1/4]
def cascade_helper(gray):
    face_cascade = cv2.CascadeClassifier('./util/model/haarcascade_frontalface_default.xml')
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(100, 100)
    )


# Helper function for detect_emotion [2/4]
def crop_face(gray, face):
    x, y, width, height = face
    return gray[y:y + height, x:x + width]


# Helper function for detect_emotion [3/4]
def preprocess_image(image):
    image = cv2.resize(image, (48, 48))
    image = image.reshape([-1, 48, 48, 1])
    image = np.multiply(image, 1.0 / 255.0)
    return image


# Helper function for detect_emotion [4/4]
def get_emotion_probabilities(image):
    emotion_model = load_model('./util/model/emotion_recognition.h5')
    probab = emotion_model.predict(image)[0] * 100
    return probab


# Function for detecting emotion
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # TODO: do we need to use this one? can't we just use the face detected by dlib? in the face_detection function
    faces = cascade_helper(gray)
    if not faces.any():
        return

    cropped_face = crop_face(gray, faces[0])
    test_image = preprocess_image(cropped_face)

    probab = get_emotion_probabilities(test_image)
    emotion = np.argmax(probab)
    emotion = emotion

    return emotion


def get_ci(gaze_weights, emotion):
    # TODO give the necessary parameters
    ci = gen_concentration_index(gaze_weights, emotion)
    return ci


# Seperate Gaze function from generate ci
# TODO

def gen_concentration_index(gaze_weights, emotion):
    # Concentration index is a percentage : max weights product = 4.5
    # dictionary for emotions
    emotions_weights = {0: 0.25, 1: 0.3, 2: 0.6, 3: 0.3, 4: 0.6, 5: 0.9}
    concentration_index = (emotions_weights[emotion] * gaze_weights) / 4.5

    if concentration_index > 0.65:
        return "You are highly focused!"
    elif 0.25 < concentration_index <= 0.65:
        return "You are focused."
    else:
        return "Pay attention!"


def process_ci(cis):
    # TODO: this is a bad solution we should find a better one
    if len(cis) > 40:
        if cis[-40:] == ["Pay attention!"] * 40:
            return "Pay attention!"
        elif cis[-20:] == ["Pay attention!"] * 20:
            return "Distracted!"
        else:
            for ci in cis[::-1]:
                if ci != "Pay attention!":
                    return ci
    elif len(cis) > 20:
        if cis[-20:] == ["Pay attention!"] * 20:
            return "Pay attention!"
        else:
            for ci in cis[::-1]:
                if ci != "Pay attention!":
                    return ci
    else:
        for ci in cis[::-1]:
            if ci != "Pay attention!":
                return ci
