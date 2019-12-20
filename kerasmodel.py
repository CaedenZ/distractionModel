from analysis import analysis
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ana = analysis()

while True:
    _, frame = cap.read()
    bm = ana.detect_face(frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()