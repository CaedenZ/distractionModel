# Program constructs Concentration Index and returns a classification of engagement.

import cv2
import numpy as np
import dlib
from math import hypot
from keras.models import load_model


class analysis:

    # Initialise models
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
        self.size = 0
        self.frame_count = 0
        self.cis = []
        self.frame_width = frame_width
        self.frame_height = frame_height

    # Function for finding midpoint of 2 points
    def midpoint(self, p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    # Function for eye size
    def get_blinking_ratio(self, frame, eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(
            eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(
            eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = self.midpoint(facial_landmarks.part(
            eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = self.midpoint(facial_landmarks.part(
            eye_points[5]), facial_landmarks.part(eye_points[4]))
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        hor_line_lenght = hypot(
            (left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_lenght = hypot(
            (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        ratio = ver_line_lenght / hor_line_lenght
        return ratio

    # Gaze detection function
    def get_gaze_ratio(self, frame, eye_points, facial_landmarks, gray):

        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(
                                        eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(
                                        eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x,
                                     facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x,
                                     facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                                   np.int32)

        height, width, _ = frame.shape
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
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        up_side_threshold = threshold_eye[0: int(height / 2), 0: int(width / 2)]
        up_side_white = cv2.countNonZero(up_side_threshold)
        down_side_threshold = threshold_eye[int(height / 2): height, 0: width]
        down_side_white = cv2.countNonZero(down_side_threshold)
        lr_gaze_ratio = (left_side_white + 10) / (right_side_white + 10)
        ud_gaze_ratio = (up_side_white + 10) / (down_side_white + 10)
        return lr_gaze_ratio, ud_gaze_ratio

    # Main function for analysis

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        faces = self.detector(gray)
        benchmark = []
        for face in faces:
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            f = gray[x:x1, y:y1]
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            landmarks = self.predictor(gray, face)
            left_point = (landmarks.part(36).x, landmarks.part(36).y)
            right_point = (landmarks.part(39).x, landmarks.part(39).y)
            center_top = self.midpoint(landmarks.part(37), landmarks.part(38))
            center_bottom = self.midpoint(
                landmarks.part(41), landmarks.part(40))
            hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
            ver_line = cv2.line(frame, center_top,
                                center_bottom, (0, 255, 0), 2)
            left_eye_ratio = self.get_blinking_ratio(frame,
                                                     [36, 37, 38, 39, 40, 41], landmarks)

            gaze_ratio_lr, gaze_ratio_ud = self.get_gaze_ratio(frame,
                                                               [36, 37, 38, 39, 40, 41], landmarks, gray)

            benchmark.append([gaze_ratio_lr, gaze_ratio_ud, left_eye_ratio])
            emotion = self.detect_emotion(gray)
            ci = self.gen_concentration_index()
            # ci is either: "You are highly engaged!", "You are engaged.", or "Pay attention!"

            self.cis.append(ci)

            # This function does our 20/10 frames logic
            ci = self.process_ci()

            # cv2.putText(frame, "x: "+str(gaze_ratio_lr),
            #             (50, 100), font, 2, (0, 0, 255), 3)
            # cv2.putText(frame, "y: "+str(gaze_ratio_ud),
            #             (50, 150), font, 2, (0, 0, 255), 3)
            # cv2.putText(frame, "Eye Size: "+str(left_eye_ratio),
            #             (50, 200), font, 2, (0, 0, 255), 3)
            emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                        3: 'Sad', 4: 'Surprised', 5: 'Neutral'}
            # if emotion:
            # cv2.putText(frame, emotions[self.emotion],
            #             (50, 150), font, 2, (0, 0, 255), 3)
            ci_to_color = {
                "You are highly focused!": (102, 255, 102),
                "You are focused.": (255, 128, 0),
                "Distracted!": (102, 178, 255),
                "Pay attention!": (0, 0, 255)

            }
            cv2.putText(
                frame, ci,
                # (50, 250), font, 2, (0, 0, 255), 3,
                (20, 100), font, 2, ci_to_color.get(ci, (0, 0, 0)), 3
            )
            # higher x value shifts to the right and lower y value shifts up.
            # Also, the number after the word "font" is the font size
            self.x = gaze_ratio_lr
            self.y = gaze_ratio_ud
            self.size = left_eye_ratio
        return frame

    def process_ci(self):
        # Look at the last 20 concentration indices and if they are all "Pay attention!",
        # then the current concentration index is "Pay attention!", otherwise if the last 10 concentration indices
        # are all "Pay attention!",
        # then the current concentration index is "Distracted!", otherwise the current concentration index is
        # last different concentration index
        if len(self.cis) > 15:
            if self.cis[-15:] == ["Pay attention!"] * 15:
                return "Pay attention!"
            elif self.cis[-7:] == ["Pay attention!"] * 7:
                return "Distracted!"
            else:
                for ci in self.cis[::-1]:
                    if ci != "Pay attention!":
                        return ci

    # Function for detecting emotion
    def detect_emotion(self, gray):
        # Dictionary for emotion recognition model output and emotions
        emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                    3: 'Sad', 4: 'Surprised', 5: 'Neutral'}

        # Face detection takes approx 0.07 seconds
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100), )
        if len(faces) > 0:
            for x, y, width, height in faces:
                cropped_face = gray[y:y + height, x:x + width]
                test_image = cv2.resize(cropped_face, (48, 48))
                test_image = test_image.reshape([-1, 48, 48, 1])

                test_image = np.multiply(test_image, 1.0 / 255.0)

                # Probablities of all classes
                # Finding class probability takes approx 0.05 seconds
                if self.frame_count % 5 == 0:
                    probab = self.emotion_model.predict(test_image)[0] * 100
                    # print("--- %s seconds ---" % (time.time() - start_time))

                    # Finding label from probabilities
                    # Class having highest probability considered output label
                    label = np.argmax(probab)
                    probab_predicted = int(probab[label])
                    predicted_emotion = emotions[label]
                    self.frame_count = 0
                    self.emotion = label

        self.frame_count += 1

        # # 	Weights from Sharma et.al. (2019)
        # Neutral	0.9
        # Happy 	0.6
        # Surprised	0.6
        # Sad	    0.3

        # Anger	    0.25
        # Fearful	0.3
        # 0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Surprised', 5: 'Neutral'}

    def gen_concentration_index(self):
        weight = 0
        emotionweights = {0: 0.25, 1: 0.3, 2: 0.6,
                          3: 0.3, 4: 0.6, 5: 0.9}

        # 	      Open Semi Close
        # Centre	5	1.5	0
        # Upright	2	1.5	0
        # Upleft	2	1.5	0
        # Right	    2	1.5	0
        # Left	    2	1.5	0
        # Downright	2	1.5	0
        # Downleft	2	1.5	0
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
        concentration_index = (emotionweights[self.emotion] * gaze_weights) / 4.5

        if concentration_index > 0.65:
            return "You are highly focused!"
        elif concentration_index > 0.25 and concentration_index <= 0.65:
            return "You are focused."
        else:
            return "Pay attention!"
