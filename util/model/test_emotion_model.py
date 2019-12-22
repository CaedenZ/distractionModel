import cv2
import numpy as np
from keras.models import load_model
import time
import sys


def web_cam(face_detector, model, src=0, vid_rec=False):
    """
    Function for recognizing emotions in real time.
    Change src = 1, if you are using external camera as
    a source of image.
    """
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Can't start camera")
        sys.exit(0)

    # Face detection
    faceCascade = face_detector
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Dictionary for emotion recognition model output and emotions
    emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                3: 'Sad', 4: 'Surprised', 5: 'Neutral'}

    # Upload images of emojis from emojis folder
    emoji = []
    for index in range(6):
        emotion = emotions[index]
        emoji.append(cv2.imread('./emojis/' + emotion + '.png', -1))

    # Code for recording real time video
    # Will record video if explicitly stated
    # Make vid_rec = True to record video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('emotion_rec1.avi', fourcc, 8.0, (640, 480))

    frame_count = 0

    while 1:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if image is being captured from source
        if not ret:
            print("No image from source")
            break

        # Convert RGB image to gray for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection takes approx 0.07 seconds
        start_time = time.time()
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100),)
        #print("--- %s seconds ---" % (time.time() - start_time))

        # Writing emotions in frame
        y0 = 15
        for index in range(6):
            cv2.putText(frame, emotions[index] + ': ', (5, y0), font,
                        0.4, (255, 0, 255), 1, cv2.LINE_AA)
            y0 += 15

        try:
            # Flag for showing probability graph of only one face
            FIRSTFACE = True
            if len(faces) > 0:
                for x, y, width, height in faces:
                    cropped_face = gray[y:y + height, x:x + width]
                    test_image = cv2.resize(cropped_face, (48, 48))
                    test_image = test_image.reshape([-1, 48, 48, 1])

                    test_image = np.multiply(test_image, 1.0 / 255.0)

                    # Probablities of all classes
                    # Finding class probability takes approx 0.05 seconds
                    start_time = time.time()
                    if frame_count % 5 == 0:
                        probab = model.predict(test_image)[0] * 100
                        #print("--- %s seconds ---" % (time.time() - start_time))

                        # Finding label from probabilities
                        # Class having highest probability considered output label
                        label = np.argmax(probab)
                        probab_predicted = int(probab[label])
                        predicted_emotion = emotions[label]
                        frame_count = 0

                    frame_count += 1
                    # Drawing probability graph for first detected face
                    if FIRSTFACE:
                        y0 = 8
                        for score in probab.astype('int'):
                            cv2.putText(frame, str(score) + '% ', (80 + score, y0 + 8),
                                        font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.rectangle(frame, (75, y0), (75 + score, y0 + 8),
                                          (0, 255, 255), cv2.FILLED)
                            y0 += 15
                            FIRSTFACE = False

                    # Drawing on frame
                    font_size = width / 300
                    filled_rect_ht = int(height / 5)

                    # Resizing emoji according to size of detected face
                    emoji_face = emoji[(label)]
                    emoji_face = cv2.resize(
                        emoji_face, (filled_rect_ht, filled_rect_ht))

                    # Positioning emojis on frame
                    emoji_x1 = x + width - filled_rect_ht
                    emoji_x2 = emoji_x1 + filled_rect_ht
                    emoji_y1 = y + height
                    emoji_y2 = emoji_y1 + filled_rect_ht

                    # Drawing rectangle and showing output values on frame
                    cv2.rectangle(frame, (x, y), (x + width,
                                                  y + height), (155, 155, 0), 2)
                    cv2.rectangle(frame, (x-1, y+height), (x+1 + width, y + height+filled_rect_ht),
                                  (155, 155, 0), cv2.FILLED)
                    cv2.putText(frame, predicted_emotion+' ' + str(probab_predicted)+'%',
                                (x, y + height + filled_rect_ht-10), font, font_size, (255, 255, 255), 1, cv2.LINE_AA)

                    # Showing emoji on frame
                    for c in range(0, 3):
                        frame[emoji_y1:emoji_y2, emoji_x1:emoji_x2, c] = emoji_face[:, :, c] * \
                            (emoji_face[:, :, 3] / 255.0) + frame[emoji_y1:emoji_y2, emoji_x1:emoji_x2, c] * \
                            (1.0 - emoji_face[:, :, 3] / 255.0)

        except Exception as error:
            # print(error)
            pass

        cv2.imshow('frame', frame)

        if vid_rec:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Creating objects for face and emotiction detection
    face_detector = cv2.CascadeClassifier(
        './model/haarcascade_frontalface_default.xml')
    emotion_model = load_model('./emotion_recognition.h5')
    web_cam(face_detector, emotion_model)


if __name__ == '__main__':
    main()
