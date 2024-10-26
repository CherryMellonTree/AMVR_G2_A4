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
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters_create()

# initialize the video file stream
print("[INFO] accessing video stream...")
vf = cv2.VideoCapture(args["input"])

# initialize a queue to maintain the next frame from the video stream
Q = deque(maxlen=128)
# we need to have a frame in our queue to start our augmented reality
# pipeline, so read the next frame from our video file source and add
# it to our queue
(grabbed, source) = vf.read()
Q.appendleft(source)
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# loop over the frames from the video stream
while len(Q) > 0:
	# grab the frame from our video stream and resize it
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	# attempt to find the ArUCo markers in the frame, and provided
	# they are found, take the current source image and warp it onto
	# input frame using our augmented reality technique
	warped = find_and_warp(
		frame, source,
		cornerIDs=(923, 1001, 241, 1007),
		arucoDict=arucoDict,
		arucoParams=arucoParams,
		useCache=args["cache"] > 0)