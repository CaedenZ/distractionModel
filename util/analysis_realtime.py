# imports
import cv2
import numpy as np
import dlib
from math import hypot
from keras.models import load_model

class analysis:
    # Global values
    # values for eye positions
    eye_points = [36, 37, 38, 39, 40, 41]
    # dictionary for emotions
    emotions_dictionary = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Surprised', 5: 'Neutral'}
    emotions_weights = {0: 0.25, 1: 0.3, 2: 0.6, 3: 0.3, 4: 0.6, 5: 0.9}

    def __init__(self, frame_width, frame_height):
        self.emotion_model = load_model('./util/model/emotion_recognition.h5')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "./util/model/shape_predictor_68_face_landmarks.dat")
        self.faceCascade = cv2.CascadeClassifier(
            './util/model/haarcascade_frontalface_default.xml')
        self.x = 0
        self.y = 0
        self.emotion = 5
        self.eye_ratio = 0
        self.frame_count = 0
        self.cis = []
        self.frame_width = frame_width
        self.frame_height = frame_height

    def get_midpoint(self, p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)
            self.display_messages()
        
        # Why return frame? maybe unnecessary
        # TODO
        return frame, landmarks


    # Function for detecting emotion

    # Should take in a frame and return the emotion. Replace gray with frame?
    # TODO
    def detect_emotion(self, gray):

        faces = self.cascade_helper(gray)
        if not faces:
            return

        cropped_face = self.crop_face(gray, faces[0])
        test_image = self.preprocess_image(cropped_face)

        probab = self.get_emotion_probabilities(test_image)
        label = np.argmax(probab)

        self.update_emotion(label)


    def display_messages(self):
        ci = self.gen_concentration_index()
        self.cis.append(ci)
        ci = self.process_ci()
        return

    # Seperate Gaze function from generate ci
    # TODO
    def gen_concentration_index(self):
        gaze_weights = 0
        if self.size < 0.2:
            gaze_weights = 0
        elif self.size > 0.2 and self.size < 0.3:
            gaze_weights = 1.5
        else:
            if self.x < 2 and self.x > 1:
                gaze_weights = 5
            else:
                gaze_weights = 2

        # Concentration index is a percentage : max weights product = 4.5
        concentration_index = (self.emotions_weights[self.emotions_dictionary] * gaze_weights) / 4.5

        if concentration_index > 0.65:
            return "You are highly focused!"
        elif concentration_index > 0.25 and concentration_index <= 0.65:
            return "You are focused."
        else:
            return "Pay attention!"


    def process_ci(self):
        if len(self.cis) > 40:
            if self.cis[-40:] == ["Pay attention!"] * 40:
                return "Pay attention!"
            elif self.cis[-20:] == ["Pay attention!"] * 20:
                return "Distracted!"
            else:
                for ci in self.cis[::-1]:
                    if ci != "Pay attention!":
                        return ci

    # Helper function for detect_emotion [1/4]
    def cascade_helper(self, gray):
        return self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100)
        )
    
    # Helper function for detect_emotion [2/4] 
    def crop_face(self, gray, face):
        x, y, width, height = face
        return gray[y:y + height, x:x + width]

    # Helper function for detect_emotion [3/4] 
    def preprocess_image(self, image):
        image = cv2.resize(image, (48, 48))
        image = image.reshape([-1, 48, 48, 1])
        image = np.multiply(image, 1.0 / 255.0)
        return image

    # Helper function for detect_emotion [4/4] 
    def get_emotion_probabilities(self, image):
        if self.frame_count % 5 != 0:
            return self.last_probab
        probab = self.emotion_model.predict(image)[0] * 100
        self.last_probab = probab
        return probab
    
    def tester (self):
        return "hi"