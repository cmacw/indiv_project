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
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*5,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('Calibration/*.jpg')
i = 1
for fname in images:
    t0 = time.time()

    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (5,5),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (5,5), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

    t1 = time.time()
    print("{}: Image {} takes {} second.".format(i, fname, t1-t0))
    i = i + 1

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savetxt("mtx.csv", mtx, delimiter=',')
print(mtx)


img_width = 4032
img_height = 3024

f_x = mtx[0,0]
f_y = mtx[1,1]
a_x = np.rad2deg(np.arctan(f_x / img_width) *2)
a_y = np.rad2deg(np.arctan(f_y / img_height) *2)
print("Fov X: {},  Y: {}".format(a_x, a_y))
