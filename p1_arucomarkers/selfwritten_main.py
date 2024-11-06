import cv2 as cv
import numpy as np
import sys

def detect_arucos(frame):
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
    if(ids is not None and len(ids) == 4):
        all_points = np.vstack([corner[0] for corner in corners])
        centroid = np.mean(all_points, axis=0)

        most_distant_points = []
        for corner in corners:
            points = corner[0]  

            distances = np.linalg.norm(points - centroid, axis=1)

            max_dist_index = np.argmax(distances)

            most_distant_points.append(points[max_dist_index])
        most_distant_points = np.array(most_distant_points)

        i=0
        cleaned_corners = [0,0,0,0]
        cleaned_ids = []
        for i in ids:
            cleaned_ids.append(i[0])
        for i in range(len(cleaned_ids)):
            cleaned_corners[cleaned_ids[i]] = most_distant_points[i]

        return cleaned_corners, ids
    return [],[]

def overlay_image_onto_markers(source_frame, modifying_frame, corners):
    if len(corners) == 4:
        # Sort corners to maintain the correct order (top-left, top-right, bottom-right, bottom-left)
        top_left = corners[0]
        top_right = corners[1]
        bottom_left = corners[2]
        bottom_right = corners[3]

        pts_dst = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

        # Resize the modifying frame to match the target aspect ratio
        h, w, _ = modifying_frame.shape
        pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Compute the perspective transform matrix and warp the modifying_frame
        matrix, _ = cv.findHomography(pts_src, pts_dst)
        warped_frame = cv.warpPerspective(modifying_frame, matrix, (source_frame.shape[1], source_frame.shape[0]))

        # Create a mask for the overlay
        mask = np.zeros((source_frame.shape[0], source_frame.shape[1]), dtype=np.uint8)
        cv.fillConvexPoly(mask, pts_dst.astype(int), 255)

        # Convert the mask to 3 channels
        mask_3ch = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        # Multiply the warped frame and source frame by the mask
        warpedMultiplied = cv.multiply(warped_frame.astype("float"), mask_3ch.astype("float") / 255)  # Normalize mask
        imageMultiplied = cv.multiply(source_frame.astype("float"), (1.0 - mask_3ch.astype("float") / 255))

        # Combine the two results
        output = cv.add(warpedMultiplied, imageMultiplied)
        output = output.astype("uint8")
        
    return output

def usingVideoFeed():
    source_video_path = "./input_2.mp4"
    modifying_video_path = "./filler_2.mp4"


    source_video_path = "./IMG_4010.mp4"
    modifying_video_path = "./example_4.mp4"

    source_video = cv.VideoCapture(source_video_path)
    modifying_video = cv.VideoCapture(modifying_video_path)
    
    source_fps = int(source_video.get(cv.CAP_PROP_FPS))
    modifying_fps = int(modifying_video.get(cv.CAP_PROP_FPS))
    fps = min(source_fps, modifying_fps)


    source_width = int(source_video.get(cv.CAP_PROP_FRAME_WIDTH))
    source_height = int(source_video.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output_2.mp4', fourcc, fps, (source_width, source_height))

    s_success, source_frame = source_video.read()
    m_success, modifying_frame = modifying_video.read()

    last_corners = []
    while s_success:
        #detect aruco
        corners, ids = detect_arucos(source_frame)

        if not m_success:
            modifying_video.set(cv.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
            m_success, modifying_frame = modifying_video.read()

        if corners is not None and len(corners) == 4:  # Ensure 4 corners are detected
            last_corners = corners
            source_frame = overlay_image_onto_markers(source_frame, modifying_frame, corners)
        elif last_corners != []:
            source_frame = overlay_image_onto_markers(source_frame, modifying_frame, last_corners)
            last_corners = []

            

        #add frame to output & get next frame if possible
        out.write(source_frame)
        s_success, source_frame = source_video.read()
        m_success, modifying_frame = modifying_video.read()

    source_video.release()
    modifying_video.release()
    out.release()
    
def usingLiveWebcamFeed():
    modifying_video_path = "./output_2.mp4"
    modifying_video = cv.VideoCapture(modifying_video_path)

    # Access the default webcam
    source_video = cv.VideoCapture(0)
    fps = int(modifying_video.get(cv.CAP_PROP_FPS))
    

    source_width = int(source_video.get(cv.CAP_PROP_FRAME_WIDTH))
    source_height = int(source_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output_feed.mp4', fourcc, fps, (source_width, source_height))

    s_success, source_frame = source_video.read()
    m_success, modifying_frame = modifying_video.read()

    last_corners = []
    while s_success:
        # Detect aruco markers
        corners, ids = detect_arucos(source_frame)
        
        if not m_success:
            modifying_video.set(cv.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
            m_success, modifying_frame = modifying_video.read()

        if corners is not None and len(corners) == 4:  # Ensure 4 corners are detected
            last_corners = corners
            source_frame = overlay_image_onto_markers(source_frame, modifying_frame, corners)
        elif last_corners != []:
            source_frame = overlay_image_onto_markers(source_frame, modifying_frame, last_corners)
            last_corners = []

        cv.imshow("Webcam Feed", source_frame)
        # Add frame to output & get next frame if possible
        out.write(source_frame)

        s_success, source_frame = source_video.read()
        m_success, modifying_frame = modifying_video.read()

        # Exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    source_video.release()
    modifying_video.release()
    out.release()

    cv.destroyAllWindows()

def main():
    #Use for AR using the given video stream
    #usingVideoFeed()
    
    #Use for AR using live webcam feed
    usingLiveWebcamFeed()

if __name__ == "__main__":
    main()