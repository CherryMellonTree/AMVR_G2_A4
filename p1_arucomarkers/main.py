# import the necessary packages
from support import find_and_warp
from imutils.video import VideoStream
from collections import deque
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video file for augmented reality")

ap.add_argument("-c", "--cache", type=int, default=-1,
	help="whether or not to use reference points cache")

args = vars(ap.parse_args())
# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] initializing marker detector...")
arucoDict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_100, 1)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# initialize the video file stream
print("[INFO] accessing video stream...")
vf = cv2.VideoCapture("IMG_4010.mp4")
Q = deque(maxlen=128)

# we need to have a frame in our queue to start our augmented reality pipeline
(grabbed, source) = vf.read()
if grabbed:
    Q.appendleft(source)

# loop over the frames from the video file
while grabbed:
    # grab the frame from our video file and resize it
    (grabbed, frame) = vf.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width=600)

    # attempt to find the ArUCo markers in the frame and apply augmented reality
    warped = find_and_warp(
        frame, source, detector,
        cornerIDs=(923, 1001, 241, 1007),
        arucoDict=arucoDict,
        arucoParams=arucoParams,
        useCache=args["cache"] > 0
    )

    # display the output frame
    cv2.imshow("Augmented Reality", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the video file and close any open windows
vf.release()
cv2.destroyAllWindows()