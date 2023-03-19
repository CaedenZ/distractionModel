# Run this file on CV2 in local machine to construct a Concentration Index (CI).
# Video image will show emotion on first line, and engagement on second. Engagement/concentration classification displays either 'Pay attention', 'You are engaged' and 'you are highly engaged' based on CI. Webcam is required.
# Analysis is in 'Util' folder.
import os.path
from playsound import playsound
from util.analysis_realtime import analysis
import cv2
import numpy as np
import argparse
from typing import List, Optional, Union
import simpleaudio as sa


# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--video_path', type=Optional[Union[List[str], str]], required=False, default=None)
parser.add_argument('--run_background', type=bool, required=False, default=False, help='Run the program in background')

# Parse the argument
args = parser.parse_args()


def run(video_path=None):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # get video name
    if video_path is None:
        video_name = 'webcam'
    else:
        video_name = os.path.basename(video_path)
        # get video name without file extension
        video_name = os.path.splitext(video_name)[0]

    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    out = cv2.VideoWriter(f'{video_name}_recorded_video.avi', fourcc, 5.0, size)

    out_with_result = cv2.VideoWriter(f'{video_name}_recorded_video_with_result.avi', fourcc, 5.0, size)
    ana = analysis(frame_width=frame_width, frame_height=frame_height)
    # Capture every frame and send to detector
    while True:
        _, frame = cap.read()
        out.write(frame)
        frame, _ = ana.detect_face(frame)

        cv2.imshow("Frame", frame)
        out_with_result.write(frame)
        key = cv2.waitKey(1)
    # Exit if 'q' is pressed
        if key == ord('q'):
            # Release the memory
            cap.release()
            out.release()
            out_with_result.release()
            cv2.destroyAllWindows()
            break

def run_background():
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    ana = analysis(frame_width=frame_width, frame_height=frame_height)
    # wave_obj = sa.WaveObject.from_wave_file("mixkit-magic-notification-ring-2344.wav")

    # Capture every frame and send to detector
    while True:
        _, frame = cap.read()
        _, ci = ana.detect_face(frame)
        if ci == "Pay attention!":
            playsound("mixkit-magic-notification-ring-2344.wav")
            # play_obj = wave_obj.play()
            # play_obj.wait_done()


        key = cv2.waitKey(1)
        if key == ord('q'):
            # Release the memory
            cap.release()
            break


if args.run_background:
    run_background()
if args.video_path is not None:
    if isinstance(args.video_path, str):
        args.video_path = [args.video_path]
    for video_path in args.video_path:
        run(video_path)
else:
    run()

