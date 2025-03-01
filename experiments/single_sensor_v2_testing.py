from stereo_calibration import StereoCameraCalibrator, undistort
from dual_sensor_testing import select_test_correspondences
from localization import triangulate, colors_list
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import lib.vxcam as vxcam
import vmbpy
import time
import os
import pickle

with open('v2_single_stereo_cam_params_take3.pkl', 'rb') as f:
    stereo_cam_params = pickle.load(f)

test_left_points, test_right_points, left_test_img, right_test_img = select_test_correspondences("box_tests/single_sensor_v2/take3_moved/bottom/LEFT/40.png", 
                                                                                                 "box_tests/single_sensor_v2/take3_moved/bottom/RIGHT/40.png", 
                                                                                                 stereo_cam_params, 
                                                                                                 undistort_frames = True)
for i, uv1 in enumerate(test_left_points):
    cv2.circle(left_test_img, tuple(uv1), 5, colors_list[i % len(colors_list)], -1)
for i, uv2 in enumerate(test_right_points):
    cv2.circle(right_test_img, tuple(uv2), 5, colors_list[i % len(colors_list)], -1)
display = np.hstack((left_test_img, right_test_img))
cv2.imshow('Selected points', display)

p3ds = triangulate(test_left_points, test_right_points, stereo_cam_params, display = True)
cv2.destroyAllWindows()