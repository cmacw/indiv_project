"""
Code Obtained From: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
Author: OpenCV
Date accessed: 28 Aug 2019
Note: The image use has 8 squares on each side, 7 internal points, which span on 148 mm thus each square has length 148/8
"""

import numpy as np
import cv2 as cv
import glob
import time

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
corner_x = 6
corner_y = 5
objp = np.zeros((corner_y * corner_x, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_y, 0:corner_x].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('Calibration/Webcam/*.png')
i = 1
success = 0
for fname in images:
    t0 = time.time()

    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (corner_y, corner_x), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (corner_y, corner_x), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
        success += 1

    t1 = time.time()
    print("{}: Image {} takes {} second.".format(i, fname, t1 - t0))
    i = i + 1

cv.destroyAllWindows()

print("Success {} / {}".format(success, len(images)))
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savetxt("mtx.csv", mtx, delimiter=',')
print(mtx)

img_width = 640
img_height = 480

f_x = mtx[0, 0]
f_y = mtx[1, 1]
a_x = np.rad2deg(np.arctan(img_width / 2 / f_x) * 2)
a_y = np.rad2deg(np.arctan(img_height / 2 / f_y) * 2)
print("Fov X: {},  Y: {}".format(a_x, a_y))
