import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from moviepy.editor import VideoFileClip

# Define a class to receive the characteristics of each line detection
class Line(): ## TODO Check what fieldds I need
    def __init__(self, current_fit):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = current_fit
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


def load_test_images_names():
    # Get the image names
    image_names = os.listdir("test_images/")
    image_full_names = []
    for img_name in image_names:
        image_full_names.append('test_images/' + img_name)
    return image_full_names

def load_image(image_name):
    return cv2.imread(image_name)

def save_image(image,  image_name):
    print "Saving image"
    image_name = image_name[12:]
    cv2.imwrite(os.path.join("output_images/", image_name), image)


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

def undistort(image, distortion_coefficients):
    print "Undistorting image "

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

def detect_lines(image, left_lines, right_lines):
    print "Detecting lines"

    if (left_lines ==[]) and (right_lines == []):
        out_img, left_line, right_line = detect_lines_windows(image)
    else:
        out_img, left_line, right_line = detect_lines_polinomial(image, left_lines, right_lines)

    left_lines.append(Line(left_line))
    right_lines.append(Line(right_line))
    return out_img

def detect_lines_polinomial(image, left_lines, right_lines):

    # Get the zone I need to look into
    poly_left_average = get_average_over_last_10_lines(left_lines)
    poly_right_average = get_average_over_last_10_lines(right_lines)
    margin = 100

    # Get the highlighted pixels
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Get the search highlighted pixels in the search area
    left_lane_inds = ((nonzerox > (poly_left_average[0]*(nonzeroy**2) + poly_left_average[1]*nonzeroy +
                    poly_left_average[2] - margin)) & (nonzerox < (poly_left_average[0]*(nonzeroy**2) +
                    poly_left_average[1]*nonzeroy + poly_left_average[2] + margin)))
    right_lane_inds = ((nonzerox > (poly_right_average[0]*(nonzeroy**2) + poly_right_average[1]*nonzeroy +
                    poly_right_average[2] - margin)) & (nonzerox < (poly_right_average[0]*(nonzeroy**2) +
                    poly_right_average[1]*nonzeroy + poly_right_average[2] + margin)))

    # Fit a polynomial on it
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    if (leftx == []) or (lefty == []): ## TODO that works only if the current fit is not too old... Add logic for the case where it is old, use the window call
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = left_lines[-1].current_fit

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    if (rightx == []) or (righty == []):
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = right_lines[-1].current_fit

    # call draw_lines on image
    out_img = draw_lines(image, left_fit, right_fit)
    return out_img, left_fit, right_fit

def get_average_over_last_10_lines(lines):
    poly_2 = []
    poly_1 = []
    poly_0 = []
    if len(lines) < 10:
        for line in lines:
            poly = line.current_fit
            poly_0.append(poly[0])
            poly_1.append(poly[1])
            poly_2.append(poly[2])
    else:
        for line in lines[-10:]:
            poly = line.current_fit
            poly_0.append(poly[0])
            poly_1.append(poly[1])
            poly_2.append(poly[2])

    return np.average(poly_0), np.average(poly_1), np.average(poly_2)


def detect_lines_windows(image):
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

    return out_img, left_fit, right_fit


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
def process_test_images():
    image_names = load_test_images_names()
    for image_name in image_names:
        print "Starting the process on image" + image_name
        distorted_image = load_image(image_name)

        result = process_one_image(distorted_image, True)

        save_image(result, image_name)


def process_one_image(image, left_lines, right_lines):
    undistorted_image = undistort(image, distortion_coefficients)
    threshold_image = apply_thresholds(undistorted_image)
    perspective_image = perspective_transform(threshold_image, False)
    lines_img = detect_lines(perspective_image, left_lines, right_lines)

    # Put back into the original space
    perspective_lines_img = perspective_transform(lines_img, True)
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, perspective_lines_img, 0.3, 0)
    return result

def process_video(video_full_name):
    left_lines = []
    right_lines = []

    clip1 = VideoFileClip(video_full_name)
    output_clip = clip1.fl_image(lambda image: process_one_image(image, left_lines, right_lines))
    output_clip.write_videofile("ouput.mp4", audio=False)


distortion_coefficients = calibrate()
# process_test_images()
process_video('harder_challenge_video.mp4')



    # TODO calculate the turn
    #TODO make it work on a video
