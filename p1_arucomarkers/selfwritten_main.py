import cv2 as cv
import numpy as np
import sys

def detect_arucos(frame):
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters_create()
    
    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    return corners, ids

def overlay_image_onto_markers(source_frame, modifying_frame, corners):
    if len(corners) == 4:
        # Sort corners to maintain the correct order (top-left, top-right, bottom-right, bottom-left)
        pts_dst = np.array([corner[0][0] for corner in corners], dtype="float32")
        pts_dst = sort_corners(pts_dst)

        # Resize the modifying frame to match the target aspect ratio
        h, w, _ = modifying_frame.shape
        pts_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

        # Compute the perspective transform matrix and warp the modifying_frame
        matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
        warped_frame = cv.warpPerspective(modifying_frame, matrix, (source_frame.shape[1], source_frame.shape[0]))

        # Create a mask for the overlay
        mask = np.zeros((source_frame.shape[0], source_frame.shape[1]), dtype=np.uint8)
        cv.fillConvexPoly(mask, pts_dst.astype(int), 255)

        # Combine the frames using the mask
        source_frame[mask == 255] = warped_frame[mask == 255]

def sort_corners(corners):
    # Sort corners into top-left, top-right, bottom-right, bottom-left
    corners = corners[np.argsort(corners[:, 1])]  # Sort by y-coordinates (top vs bottom)
    top_corners = corners[:2]
    bottom_corners = corners[2:]
    top_left, top_right = top_corners[np.argsort(top_corners[:, 0])]
    bottom_left, bottom_right = bottom_corners[np.argsort(bottom_corners[:, 0])]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

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
        
        if corners is not None and len(corners) == 4:  # Ensure 4 corners are detected
            overlay_image_onto_markers(source_frame, modifying_frame, corners)

        #add frame to output & get next frame if possible
        out.write(source_frame)
        s_success, source_frame = source_video.read()
        m_success, modifying_frame = modifying_video.read()

    source_video.release()
    modifying_video.release()
    out.release()

if __name__ == "__main__":
    main()