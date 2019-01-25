import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from moviepy.editor import VideoFileClip

## Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, current_fit, allx, ally):
        self.current_fit = current_fit
        #x values for detected line pixels
        self.allx = np.float_(allx)
        #y values for detected line pixels
        self.ally = np.float_(ally)
        self.curv = None

## Load all the image names including the path
## Returns and array of names
def load_test_images_names():
    # Get the image names
    image_names = os.listdir("test_images/")
    image_full_names = []
    for img_name in image_names:
        image_full_names.append('test_images/' + img_name)
    return image_full_names

## Loads an image given an image full name
def load_image(image_name):
    return cv2.imread(image_name)

## Saves an image with the same name as the original image in the output_images folder
def save_image(image,  image_name):
    print "Saving image"
    image_name = image_name[12:]
    cv2.imwrite(os.path.join("output_images/", image_name), image)

## Gets the camera distortion coefficients and returns them.
# 1- Load calibration pictures
# 2- Convert then to gray scale
# 3- Get the chessboard corners
# 4-  Get the camera coefficients, and return them
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

    return ret, mtx, dist, rvecs, tvecs

## Undistorts the original image using distortion coefficients
def undistort(image, distortion_coefficients):
    print "Undistorting image "

    mtx = distortion_coefficients[1]
    dist = distortion_coefficients[2]

    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)

    return undistorted_image

## Applies gradient and color thresholds
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

    # Convert to HLS
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls_image[:, :, 2]
    hls_binary = np.zeros_like(s_channel)
    hls_binary[(s_channel > 200) & (s_channel < 255)] = 1

    luv_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(luv_image[:,:,0])
    ax1.set_title('L', fontsize=30)
    ax2.imshow(luv_image[:,:,1])
    ax2.set_title('U', fontsize=30)
    ax3.imshow(luv_image[:,:,2])
    ax3.set_title('V', fontsize=30)
    plt.show()




    combined = np.zeros_like(magniture_binary)
    combined[((magniture_binary == 1) & (sx_binary == 1)) | (hls_binary == 1)] = 1

    return combined

## Transforms the image to a different perspective.
## If inverse == False, transform from original image to bird view perspective
## Else, transform from bird view to the original perspective
## Returns the new image
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

    return warped

## Detects the lanes in the bird view picuture using the right method (window sliding or based on the known previous polynomial)
def detect_lines(image, left_lines, right_lines, force_window_detection):
    print "Detecting lines"

    if force_window_detection:
        left_lines.pop()
        right_lines.pop()

    if ((left_lines ==[]) and (right_lines == []) or force_window_detection):
        left_line, right_line = detect_lines_windows(image, left_lines, right_lines)
    else:
        left_line, right_line = detect_lines_polinomial(image, left_lines, right_lines)

    left_lines.append(left_line)
    right_lines.append(right_line)

## Detects both the left and right lanes based on the last polinomial
def detect_lines_polinomial(image, left_lines, right_lines):

    # Get the zone I need to look into
    poly_left_average = get_average_over_last_n_lines(left_lines, 5)
    poly_right_average = get_average_over_last_n_lines(right_lines, 5)
    margin = 50

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
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    is_left_too_small = (len(leftx) <= 10) or (len(lefty) <= 10)
    is_right_too_small = (len(rightx) <= 10) or (len(righty) <= 10)

    if is_left_too_small or is_right_too_small:
        left_line, right_line = detect_lines_windows(image, left_lines, right_lines)

    if not is_left_too_small:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_line = Line(left_fit, leftx, lefty)

    if not is_right_too_small:
        right_fit = np.polyfit(righty, rightx, 2)
        right_line = Line(right_fit, rightx, righty)

    return left_line, right_line

## Calculates the average of the polynomial coefficients for the past n images
def get_average_over_last_n_lines(lines, n):
    poly_2 = []
    poly_1 = []
    poly_0 = []

    if len(lines) < n:
        lines_avg = lines
    else:
        lines_avg = lines[-n:]

    for line in lines_avg:
        poly = line.current_fit
        poly_0.append(poly[0])
        poly_1.append(poly[1])
        poly_2.append(poly[2])

    return np.average(poly_0), np.average(poly_1), np.average(poly_2)

## Detects the lanes using the sliding window method
def detect_lines_windows(image, left_lines, right_lines):
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

    left_fit = []
    right_fit = []

    if (len(left_peaks_x) > 0) and (len(left_peaks_y) > 0):
        left_fit = np.polyfit(left_peaks_y, left_peaks_x, 2)
        left_line = Line(left_fit, left_peaks_x, left_peaks_y)
    else:
        left_line = left_lines[-1]

    if (len(right_peaks_x) > 0) and (len(right_peaks_y) > 0):
        right_fit = np.polyfit(right_peaks_y, right_peaks_x, 2)
        right_line = Line(right_fit, right_peaks_x, right_peaks_x)
    else:
        right_line = right_lines[-1]

    return left_line, right_line

## Draws the lane in the bird view image
## Input:
#    image: the image to  draw the lines in
#    left_fit: the polynomial coefficients of the polynomial that fits the left points
#    right_fit: the polynomial coefficients of the polynomial that fits the right points
## Output: the image with only the lines drawned in it
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

## Detects 1 lane using the window slicing method
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

    histogram = np.sum(image[window_bottom_y:window_top_y, :], axis=0)

    #  If we have more than 50 pixels, append to our peak list
    if np.int(histogram[window_left_x:window_right_x].sum()) > 25:
        left_peak = np.argmax(histogram[window_left_x:window_right_x])
        peaks_x.append(left_peak + window_left_x)
        peaks_y.append(window_top_y)

## Measure the curvature for the left and right lines
## Input:
#    left_lines:  an array of the previous images Lines
#    right_lines:  an array of the previous images Lines
## Output: the values of both the left and right curvatures
def measure_curvature(left_lines, right_lines):
    left_curv = measure_one_line_curvature(left_lines[-1])
    right_curv = measure_one_line_curvature(right_lines[-1])

    return left_curv, right_curv

## Measure the curvature of on lane
## Input: the Line object we want to measure the curvature on
## Output: the curvature value
def measure_one_line_curvature(line):
    ym_per_pix = 15. / 720
    xm_per_pix = 3.7 / 700

    allx = line.allx
    ally = line.ally

    poly = np.polyfit(ally * ym_per_pix, allx * xm_per_pix, 2)
    y_eval = np.max(ally)

    curv = ((1 + (2 * poly[0] * y_eval * ym_per_pix + poly[1]) ** 2) ** 1.5) / np.absolute(2 * poly[0])
    line.curv = curv

    return curv

## Measure the distance between the center of the car and the cenrer of the line
## Input: the image, and both Line object for the left and right linee
## Output: a string describing the position of the car relative to the center of the line
def measure_distance_from_center(image, left_line, right_line):
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit

    y_eval = image.shape[1]

    left_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]

    center_car = image.shape[1] // 2
    center_lane = (left_bottom + right_bottom) // 2

    distance_pixel = center_lane - center_car
    distance_meter = distance_pixel * 3.7 / 700
    is_left = distance_meter >= 0

    if is_left:
        distance_str = "Distance: " + str(distance_meter)[:5] + "m left of the center"
    else:
        distance_str = "Distance: " + str( -distance_meter)[:5] + "m right of the center"

    return distance_str

## Applies the full pipeline to an image
## Does a sanity check on the curvature result and reprocesses the image if necessary
## Input:
#    image:
#    left_lines:  an array of Lines representing the history of the left line
#    right_lines:  an array of Lines representing the history of the right line
## Output: The processed image

def process_one_image(image, left_lines, right_lines):
    # Undistort
    undistorted_image = undistort(image, distortion_coefficients)
    # Apply grandient and color threshold
    threshold_image = apply_thresholds(undistorted_image)
    # Convert into bird view
    perspective_image = perspective_transform(threshold_image, False)
    # Detect line
    detect_lines(perspective_image, left_lines, right_lines, False)
    # Measure the curvature
    curv = measure_curvature(left_lines, right_lines)
    print curv

    # If the curvature of the left or the right is less than 200 or more than 50 000
    # There is an issue, so we want to try and detect the lanes forcing the window sliding method
    # If the value of the curvature is still off, use the last line.
    if (curv[0] < 200) | (curv[0] > 50000):
        detect_lines(perspective_image, left_lines, right_lines, True)
        curv = measure_curvature(left_lines, right_lines)
        if ((curv[0] < 200) | (curv[0] > 50000)) & (len(left_lines) > 1):
            left_lines.pop()
            left_lines.append(left_lines[-1])

    if (curv[1] < 200) | (curv[1] > 50000):
        detect_lines(perspective_image, left_lines, right_lines, True)
        curv = measure_curvature(left_lines, right_lines)
        if ((curv[1] < 200) | (curv[1] > 50000)) & (len(right_lines) > 1):
            right_lines.pop()
            right_lines.append(right_lines[-1])


    # Draw lines only once we have the final left and right lines
    if (left_lines > 0) & (right_lines > 0):
        lines_img = draw_lines(perspective_image, left_lines[-1].current_fit, right_lines[-1].current_fit)

    # Put back into the original space
    perspective_lines_img = perspective_transform(lines_img, True)
    if (left_lines > 0) & (right_lines > 0):
        curvature = (left_lines[-1].curv + right_lines[-1].curv) / 2
        if curvature != float('inf') :
            cv2.putText(perspective_lines_img, "Curvature: " + str(int(curvature)) + "m", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        distance_text = measure_distance_from_center(image, left_lines[-1], right_lines[-1])
        cv2.putText(perspective_lines_img, distance_text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, perspective_lines_img, 0.3, 0)

    return result

## Processes all the images in the test_image folder
## And saves them in the test_image_output_folder
def process_test_images():
    image_names = load_test_images_names()
    for image_name in image_names:
        print "Starting the process on image" + image_name
        distorted_image = load_image(image_name)

        result = process_one_image(distorted_image, [], [])

        save_image(result, image_name)


## Processes all the images of a video
## And saves it in output.mp4
## Input: the full name of the video the process
## Output: None
def process_video(video_full_name):
    left_lines = []
    right_lines = []

    clip1 = VideoFileClip(video_full_name)
    output_clip = clip1.fl_image(lambda image: process_one_image(image, left_lines, right_lines))
    output_clip.write_videofile("ouput.mp4")

## Runs the whole code
distortion_coefficients = calibrate()
process_test_images()
# process_video('project_video.mp4')

