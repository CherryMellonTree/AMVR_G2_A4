import cv2
import numpy as np



if __name__ == "__main__":
    INPUT_DIR = "./inputs"

    #for actual webcam replace with
    #webcam = cv2.VideoCapture(0)
    webcam = cv2.VideoCapture(f"{INPUT_DIR}/edm_vid.mp4")
    target = cv2.imread(f"{INPUT_DIR}/edm_image.png")
    replace_img = cv2.imread(f"{INPUT_DIR}/ypsat.jpeg")

    height_target, width_target, channels_target = target.shape
    replace_img = cv2.resize(replace_img, (width_target, height_target))

    # chose nr of features = 2000 to get approx. 50 good matches with knnMatch
    # with nr of features = 1000 we only got approx. 30 good ones
    orb = cv2.ORB_create(nfeatures=2000)
    target_key_points, target_descriptors = orb.detectAndCompute(target, None)
    target = cv2.drawKeypoints(target, target_key_points, None)

    while 1 + 1 != 3: #haha
        _, webcam_frame = webcam.read()
        augmented_img = webcam_frame.copy()
        dest_key_points, dest_descriptors = orb.detectAndCompute(webcam_frame, None)
        webcam_frame = cv2.drawKeypoints(webcam_frame, dest_key_points, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(target_descriptors, dest_descriptors, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < (0.75 * n.distance):
                good_matches.append(m)
        print(f"found {len(good_matches)} good matches")

        features = cv2.drawMatches(target, target_key_points, webcam_frame, dest_key_points, good_matches, None, flags=2)

        # only do homography when enough good matches
        if len(good_matches) > 30:
            #find points of the good matches
            source_points = np.float32([target_key_points[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dest_points = np.float32([dest_key_points[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, _ = cv2.findHomography(source_points, dest_points, method=cv2.RANSAC, ransacReprojThreshold=5)
            #print(H)
            
            #finding bounding box
            shape_bounding_box = np.float32([[0, 0], [0, height_target], [width_target, height_target], [width_target, 0]]).reshape(-1, 1, 2)
            bounding_box = cv2.perspectiveTransform(shape_bounding_box, H)
            viz_bounding_box = cv2.polylines(webcam_frame, [np.int32(bounding_box)], True, color=(255,0,255), thickness=3)

            #apply homography to replacement image
            warped_replacement = cv2.warpPerspective(replace_img, H, dsize=(webcam_frame.shape[1], webcam_frame.shape[0]))
            mask = np.zeros((webcam_frame.shape[0], webcam_frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(bounding_box)], color=(255, 255, 255))
            inv_mask = cv2.bitwise_not(mask, None)
            augmented_img = cv2.bitwise_and(augmented_img, augmented_img, mask=inv_mask)
            augmented_img = cv2.bitwise_or(warped_replacement, augmented_img, mask=None)
            
        cv2.imshow("Features: Webcam vs Target", features)
        # cv2.imshow("Target", target)
        # cv2.imshow("Webcam", webcam_frame)
        #cv2.imshow("Warped", warped_replacement)
        #cv2.imshow("Bounding box", viz_bounding_box)
        #cv2.imshow("Inv", inv_mask)
        cv2.imshow("Augmented img", augmented_img)
        #cv2.imshow("Replacement IMG", replace_img)
        cv2.waitKey(1)