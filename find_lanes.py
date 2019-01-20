import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_test_images_names():
    # Get the image names
    image_names = os.listdir("test_images/")
    image_full_names = []
    for img_name in image_names:
        image_full_names.append('test_images/' + img_name)
    return image_full_names

def load_image(image_name):
    return cv2.imread(image_name)

## Load calibration pictures
## Convert then to gray scale
## Get the chessboard corners
## Get the camera coefficients, and return them
def calibrate():
    print "Calibrating the camera"
    calibration_dir = os.listdir("camera_cal/")
    pattern_size = (9, 6)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    object_points = []
    image_points = []
    for cal_img_name in calibration_dir:
        chessboard_img = cv2.imread("camera_cal/" + cal_img_name)
        gray = cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            object_points.append(objp)
            image_points.append(corners)

    img = cv2.imread("camera_cal/" + calibration_dir[0])
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size, None, None)
    print "The distortions coefficients are found"

    ## Note: Uncomment the following lines to test  on an image
    # test_img = cv2.imread("camera_cal/" + calibration_dir[0])
    # dst = cv2.undistort(test_img, mtx, dist, None, mtx)
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.imshow(test_img)
    # ax1.set_title('Original Image', fontsize=30)
    # ax2.imshow(dst)
    # ax2.set_title('Undistorted Image', fontsize=30)
    # plt.show()

    return ret, mtx, dist, rvecs, tvecs

def undistort(image_name, distortion_coefficients):
    print "Undistorting image " + image_name
    image = load_image(image_name)

    mtx = distortion_coefficients[1]
    dist = distortion_coefficients[2]

    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)

    ## Note: Uncomment the following lines to test  on an image
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.imshow(image)
    # ax1.set_title('Original Image', fontsize=30)
    # ax2.imshow(undistorted_image)
    # ax2.set_title('Undistorted Image', fontsize=30)
    # plt.show()

    return undistorted_image

def apply_thresholds(image):
    print "Applying color and gradient threshold"
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Get the scaled Sobels
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    absolute_sobelx = np.absolute(sobelx)

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    absolute_sobely = np.absolute(sobely)

    # Sobelx
    scaled_sobelx = np.uint8(255 * absolute_sobelx / np.max(absolute_sobelx))
    sx_binary = np.zeros_like(scaled_sobelx)
    sx_binary[(scaled_sobelx >= 40) & (scaled_sobelx <= 200)] = 1

    # Magnitude of the gradient
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel_mag = np.uint8(255 * magnitude / np.max(magnitude))
    magniture_binary = np.zeros_like(magnitude)
    magniture_binary[(scaled_sobel_mag >= 150) & (scaled_sobel_mag <= 300)] = 1

    # Direction of the gradient
    direction = np.arctan2(absolute_sobely, absolute_sobelx)
    direction_binary = np.zeros_like(direction)
    direction_binary[(direction >= 0.3) & (direction <= 0.7)] = 1

    # Convert to HLS
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls_image[:, :, 2]
    hls_binary = np.zeros_like(s_channel)
    hls_binary[(s_channel > 200) & (s_channel < 255)] = 1

    combined = np.zeros_like(direction)
    combined[((magniture_binary == 1) & (sx_binary == 1)) | (hls_binary == 1)] = 1

    ## Note: Uncomment the following lines to test  on an image
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.imshow(image)
    # ax1.set_title('Original', fontsize=30)
    # ax2.imshow(combined)
    # ax2.set_title('Thresholded', fontsize=30)
    # plt.show()

    return combined

def perspective_transform(image):
    print "Transforming the perspective"
    img_size = (image.shape[1], image.shape[0])
    src = np.float32([[447, 548], [844, 548], [1005, 647], [301, 647]])

    dst = np.float32([[301, 475], [1005, 475],
                      [1005, 647], [301, 647]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, img_size)

    ## Note: Uncomment the following lines to test  on an image
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.imshow(image)
    # ax1.set_title('Theshold', fontsize=30)
    # ax2.imshow(warped)
    # ax2.set_title('Warped', fontsize=30)
    # plt.show()

    return warped

def detect_lines(image):
    shape = image.shape

    #Get the bottom third of the picture
    number_of_windows = 5
    window_size = np.int(image.shape[0]//number_of_windows)

    left_peaks_y = []
    left_peaks_x = []
    right_peaks_y = []
    right_peaks_x = []

    for window_id in range(number_of_windows):

        # TODO clean this up
        left_lane_point = detect_one_lane(image, left_peaks_x, shape, window_id, window_size, True)
        left_peaks_x.append(left_lane_point[0])
        left_peaks_y.append(left_lane_point[1])

        right_lane_point = detect_one_lane(image, right_peaks_x, shape, window_id, window_size, False)
        right_peaks_x.append(right_lane_point[0])
        right_peaks_y.append(right_lane_point[1])

    left_fit = np.polyfit(left_peaks_y, left_peaks_x, 2)
    right_fit = np.polyfit(right_peaks_y, right_peaks_x, 2)
    print left_fit
    print right_fit

    #TODO draw the line

def detect_one_lane(image, left_peaks_x, shape, window_id, window_size, isLeft):
    ## Left window
    window_bottom_y = int(shape[0] - (window_id + 1) * window_size)
    window_top_y = int(shape[0] - window_id * window_size)

    if len(left_peaks_x) == 0:
        if isLeft:
            window_left_x = 0
            window_right_x = shape[1] // 2
        else:
            window_left_x = shape[1] // 2
            window_right_x = shape[1]
    else:
        window_left_x = left_peaks_x[-1] - 100
        window_right_x = left_peaks_x[-1] + 100

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.imshow(image)
    # ax1.set_title('Theshold', fontsize=30)
    # ax2.imshow(image[window_bottom_y:window_top_y, :])
    # ax2.set_title('Warped', fontsize=30)
    # plt.show()

    histogram = np.sum(image[window_bottom_y:window_top_y, :], axis=0)

    # Find the 2 peaks, where the max of the histogram is
    left_peak = np.argmax(histogram[window_left_x:window_right_x])
    return left_peak + window_left_x, window_top_y

# Run the code
distortion_coefficients = calibrate()
image_names = load_test_images_names()
for image in image_names:
    undistorted_image = undistort(image, distortion_coefficients)
    threshold_image = apply_thresholds(undistorted_image)
    perspective_image = perspective_transform(threshold_image)
    detect_lines(perspective_image)
    # TODO calculate the turn
    # TODO create an image without the perspective with the line
    # TODO Put it in the image perspective
    # TODO create a new image with both
