import cv2
import numpy as np
img = cv2.imread("Book1.jpg", cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(0)

sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)
img=cv2.drawKeypoints(img, kp_image, img)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:

    _, frame=cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    grayframe=cv2.drawKeypoints(grayframe, kp_grayframe, grayframe)

    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)

    img3=cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)

    if len(good_points)>10:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w, _ = img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        cv2.imshow("Homeography", homography)
    else:
        cv2.imshow("Homeography",grayframe)

    cv2.namedWindow('Matrix Comparison', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('Matrix Comparison', 600, 600)
    cv2.imshow('Matrix Comparison',img3)
    #cv2.imshow("Image",img)
    #cv2.imshow("GrayFrame", grayframe)
    cv2.imshow("Frame", frame)


    key=cv2.waitKey(1)
    if key== 27:
        break


cap.release()
cv2.destroyAllWindows()
