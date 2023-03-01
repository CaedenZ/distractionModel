# Run this file on CV2 in local machine to construct a Concentration Index (CI).
# Video image will show emotion on first line, and engagement on second. Engagement/concentration classification displays either 'Pay attention', 'You are engaged' and 'you are highly engaged' based on CI. Webcam is required.
# Analysis is in 'Util' folder.


from util.analysis_realtime import analysis
import cv2
import numpy as np


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('recorded_video.avi', fourcc, 20.0, (640,  480))
out_with_result = cv2.VideoWriter('recorded_video_with_result.avi', fourcc, 20.0, (640,  480))

# Initializing
cap = cv2.VideoCapture(0)
ana = analysis()

# Capture every frame and send to detector
while True:
    _, frame = cap.read()
    out.write(frame)
    bm = ana.detect_face(frame)

    cv2.imshow("Frame", frame)
    out_with_result.write(frame)
    key = cv2.waitKey(1)
# Exit if 'q' is pressed
    if key == ord('q'):
        break

# Release the memory
cap.release()
out.release()
out_with_result.release()
cv2.destroyAllWindows()
