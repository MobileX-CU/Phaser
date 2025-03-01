"""
Utility functions for image processing, including selection of test points
in the stereo camera set up and contour detection for the laser spot
"""

from stereo_calibration import undistort
from localization import colors_list

import cv2
import numpy as np
import os

def get_brightest_contour(frame, contour_area_threshold = float('inf'), otsu_adjustment = -10):
    # get contours
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_ret, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(blurred, otsu_ret + otsu_adjustment, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    # remove any that are too large
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if contour_area_threshold is not None:
            if area > contour_area_threshold:
                continue
        
        mask = np.zeros(gray.shape, np.uint8) # get brightness
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_brightness = cv2.mean(gray, mask=mask)[0]
        x, y, w, h = cv2.boundingRect(contour)
        """
        Contour object, modeling all contours as a circle

        contour_obj = {
            center : x, y (ints)
            radius : int
            brightness : int
            area = : int
        }
        """
        contour_obj     = {
            "center" : (int(x+(w/2)), int(y+(h/2))),
            "radius" : int(((w / 2) + (h/2)) / 2),
            "brightness" : mean_brightness,
            "area" : area
        }
        filtered_contours.append(contour_obj)
    
    if filtered_contours == []:
        return None
    
    # get the brightest blob
    target = max(filtered_contours, key=lambda x: x["brightness"])

    return target


def select_test_correspondences(input1_path, input2_path, stereo_cam_params, undistort_frames = True):
    selected_points = []
    curr_color_num = [-1]
   
    #click event function
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            if len(selected_points) % 2 == 0:
                selected_points.append([x,y])
                curr_color_num[0] += 1
            elif selected_points[-1][0] <= left.shape[1] and x <= left.shape[1]:
                print("Invalid correspondence. Please select a point in the right half.")
            elif selected_points[-1][0] > left.shape[1] and x > left.shape[1]:
                print("Invalid correspondence. Please select a point in the left half.")
            elif len(selected_points) % 2 == 1:
                selected_points.append([x,y])
                print("Correspondence recorded")
            else:
                print("Error recording point.")
            cv2.drawMarker(vis_img, (x,y), colors_list[curr_color_num[0]], markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.imshow("Select correspondence in both halves ([a] to accept, [q] to exit)", vis_img)
    
    # prep input for selection
    extension1 = os.path.splitext(input1_path)[-1]
    extension2 = os.path.splitext(input2_path)[-1]
    assert extension1 == extension2, "Input files must have the same extension."
    if extension1 in [".mp4", ".avi", ".mov"]:
        cap1 = cv2.VideoCapture(input1_path)
        cap2 = cv2.VideoCapture(input2_path)
        while True:
            ret1, img1 = cap1.read()
            ret2, img2 = cap2.read()
            if not ret1 or not ret2:
                break
            left = img1
            right = img2
            if undistort_frames:
                left = undistort(left, stereo_cam_params.cam1_params)
                right = undistort(right, stereo_cam_params.cam2_params)
            vis_img = np.hstack((left.copy(), right.copy()))
            vis_img = cv2.line(vis_img, (left.shape[1], 0), (left.shape[1], vis_img.shape[0]), (255, 255, 0), 2)
            cv2.imshow("Select test frame ([q] to select and exit)", vis_img)
            if cv2.waitKey(1) == ord('q'): break
        cv2.destroyAllWindows()
    else:
        left = cv2.imread(input1_path)
        right = cv2.imread(input2_path)
        if undistort_frames:
            left = undistort(left, stereo_cam_params.cam1_params)
            right = undistort(right, stereo_cam_params.cam2_params)
        vis_img = np.hstack((left.copy(), right.copy()))
        vis_img = cv2.line(vis_img, (left.shape[1], 0), (left.shape[1], vis_img.shape[0]), (255, 255, 0), 2)
       
    cv2.imshow("Select correspondence in both halves ([a] to accept, [q] to exit)", vis_img)
    cv2.setMouseCallback("Select correspondence in both halves ([a] to accept, [q] to exit)", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # subtract left.shape[1] from x coordinate of right frame points
    # assumes that right frame was displayed hstacked to the left frame
    left_frame_points = []
    right_frame_points = []
    for s in selected_points:
        if s[0] < left.shape[1]:
            left_frame_points.append(s)
        else:
            s_updated = [s[0] - left.shape[1], s[1]]
            right_frame_points.append(s_updated)

    return left_frame_points, right_frame_points, left, right