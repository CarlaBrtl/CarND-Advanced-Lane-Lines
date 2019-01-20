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
    magniture_binary[(scaled_sobel_mag >= 50) & (scaled_sobel_mag <= 300)] = 1

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

def perspective_transform(image,  inverse):
    print "Transforming the perspective"
    img_size = (image.shape[1], image.shape[0])
    src = np.float32([[447, 548], [844, 548], [1005, 647], [301, 647]])

    dst = np.float32([[301, 475], [1005, 475],
                      [1005, 647], [301, 647]])

    if inverse:
        M = cv2.getPerspectiveTransform(dst, src)
    else:
        M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

    ## Note: Uncomment the following lines to test  on an image
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.imshow(image)
    # ax1.set_title('Theshold', fontsize=30)
    # ax2.imshow(warped)
    # ax2.set_title('Warped', fontsize=30)
    # plt.show()

    return warped

def detect_lines(image):
    print "Detecting lines"
    shape = image.shape

    #Get the bottom third of the picture
    number_of_windows = 100
    window_size = np.int(image.shape[0]//number_of_windows)

    left_peaks_y = []
    left_peaks_x = []
    right_peaks_y = []
    right_peaks_x = []

    for window_id in range(number_of_windows):

        # Fill left_peaks_x and left_peaks_y with the points of the left lane
        detect_one_lane(image, left_peaks_x, left_peaks_y, shape, window_id, window_size, True)
        # Fill right_peaks_x and right_peaks_y with the points of the left lane
        detect_one_lane(image, right_peaks_x, right_peaks_y, shape, window_id, window_size, False)

    left_fit = np.polyfit(left_peaks_y, left_peaks_x, 2)
    right_fit = np.polyfit(right_peaks_y, right_peaks_x, 2)

    out_img = draw_lines(image, left_fit, right_fit)

    return out_img


def draw_lines(image, left_fit, right_fit):
    # Assuming we have `left_fit` and `right_fit` from `np.polyfit` before
    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    return color_warp

def detect_one_lane(image, peaks_x, peaks_y, shape, window_id, window_size, isLeft):
    ## Left window
    window_bottom_y = int(shape[0] - (window_id + 1) * window_size)
    window_top_y = int(shape[0] - window_id * window_size)

    if len(peaks_x) == 0:
        if isLeft:
            window_left_x = 0
            window_right_x = shape[1] // 2
        else:
            window_left_x = shape[1] // 2
            window_right_x = shape[1]
    else:
        window_left_x = peaks_x[-1] - 100
        window_right_x = peaks_x[-1] + 100

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.imshow(image)
    # ax1.set_title('Theshold', fontsize=30)
    # ax2.imshow(image[window_bottom_y:window_top_y, :])
    # ax2.set_title('Warped', fontsize=30)
    # plt.show()

    histogram = np.sum(image[window_bottom_y:window_top_y, :], axis=0)

    #  If we have more than 50 pixels, append to our peak list
    if np.int(histogram[window_left_x:window_right_x].sum()) > 50:
        left_peak = np.argmax(histogram[window_left_x:window_right_x])
        peaks_x.append(left_peak + window_left_x)
        peaks_y.append(window_top_y)


# Run the code

distortion_coefficients = calibrate()
image_names = load_test_images_names()
for image in image_names:
    print "Starting the process"
    undistorted_image = undistort(image, distortion_coefficients)
    threshold_image = apply_thresholds(undistorted_image)
    perspective_image = perspective_transform(threshold_image, False)
    lines_img = detect_lines(perspective_image)
    # Put back into the original space
    perspective_lines_img = perspective_transform(lines_img, True)

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, perspective_lines_img, 0.3, 0)


    ## Note: Uncomment the following lines to test  on an image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(undistorted_image)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(result)
    ax2.set_title('Processed image', fontsize=30)
    plt.show()

    # TODO calculate the turn


#TODO make it work on a video
