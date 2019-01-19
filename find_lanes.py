import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_test_images_names():
    # Get the image names
    image_names = os.listdir("test_images/")
    image_full_names = []
    for img_name in image_names:
        image_full_names.append(cv2.imread('test_images/' + img_name))
    return image_names

def load_image(image_name):
    return cv2.imread(image_name)

## Load calibration pictures
## Convert then to gray scale
## Get the chessboard corners
## Get the camera coefficients, and return them
def calibrate():
    print "Calibrating the camera"
    calibration_dir = os.listdir("camera_cal/")
    pattern_size = (9, 5)
    objp = np.zeros((5 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2)

    object_points = []
    image_points = []
    for cal_img_name in calibration_dir:
        chessboard_img = cv2.imread("camera_cal/" + cal_img_name)
        gray = cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret == True:
            object_points.append(objp)
            image_points.append(corners)

    img = cv2.imread("camera_cal/" + calibration_dir[0])
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size, None, None)
    print "The distortions coefficients are found"

    return ret, mtx, dist, rvecs, tvecs

    # For each of the pictures find the chessboard corners, and then draw them

def pipeline():
    dist_coeffs = calibrate()

    print dist_coeffs

    img_names = load_test_images_names()
    for img_name in img_names:
        original_image = load_image(img_name)
        # undistort(original_image)

pipeline()