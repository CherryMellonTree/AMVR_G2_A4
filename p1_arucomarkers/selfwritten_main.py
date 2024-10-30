import cv2 as cv
import numpy as np
import sys

def detect_arucos(frame):
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
    return corners, ids

def main():
    source_video_path = "./IMG_4010.mp4"
    modifying_video_path = "./BALLS.mp4"

    source_video = cv.VideoCapture(source_video_path)
    modifying_video = cv.VideoCapture(modifying_video_path)
    
    source_fps = int(source_video.get(cv.CAP_PROP_FPS))
    modifying_fps = int(modifying_video.get(cv.CAP_PROP_FPS))
    fps = min(source_fps, modifying_fps)


    source_width = int(source_video.get(cv.CAP_PROP_FRAME_WIDTH))
    source_height = int(source_video.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output.mp4', fourcc, fps, (source_width, source_height))

    s_success, source_frame = source_video.read()
    m_success, modifying_frame = modifying_video.read()

    last_valid_corners = []
    last_valid_ids = []
    while m_success and s_success:
        #detect aruco
        corners, ids = detect_arucos(source_frame)

        #inject second feed
        all_corners = np.concatenate(corners).reshape(-1, 2)
        hull = cv.convexHull(all_corners)
        hull = hull.astype(int)

        cv.polylines(source_frame, [hull], isClosed=True, color=(0, 255, 0), thickness=2)

        
        #add frame to output & get next frame if possible
        out.write(source_frame)
        s_success, source_frame = source_video.read()
        m_success, modifying_frame = modifying_video.read()

    source_video.release()
    modifying_video.release()
    out.release()

if __name__ == "__main__":
    main()