
"""
Visualize blob detection in stereo camera images.
Utility script for testing and debugging purposes, not actual operation.
"""

import lib.mirror as mirror
import lib.vxcam as vxcam
import vmbpy
from camera_settings import dual_cam_settings
from im_proc import select_test_correspondences
from stereo_calibration import *
from localization import triangulate
from im_proc import *
from laser_calibration import transform_cam2laser

import sys
import pickle
import numpy as np
import traceback
import cv2 
import time
import os
import math
import matplotlib.pyplot as plt


DISPLAY = True # dislplay stereo cam view and detetcted blobs

stereo_params_path = "params/example_stereo_params.pkl"
with open(stereo_params_path, "rb") as f:
    stereo_params = pickle.load(f)

try: 
    # set up cameras
    with vmbpy.VmbSystem.get_instance() as vmb:
            cams = vmb.get_all_cameras()
            num_cams = len(cams)
            print(f"Found {num_cams} cameras.")
            if num_cams != 2:
                print("Need min 2 cameras to run this function, exiting.")
                sys.exit(1)

    with vxcam.VXCam() as vx1, vxcam.VXCam(1) as vx2:
        
        vx1.pixel_format = dual_cam_settings["cam1"]["pixel_format"]
        vx2.pixel_format = dual_cam_settings["cam2"]["pixel_format"]
        vx1.auto_exposure = dual_cam_settings["cam1"]["auto_exposure"]
        vx2.auto_exposure = dual_cam_settings["cam2"]["auto_exposure"]
        vx1.exposure_us = dual_cam_settings["cam1"]["deploy_exposure_us"]
        vx2.exposure_us =  dual_cam_settings["cam2"]["deploy_exposure_us"]
        vx1.gain_db = dual_cam_settings["cam1"]["deploy_gain_db"]
        vx2.gain_db = dual_cam_settings["cam2"]["deploy_gain_db"]
        vx1.white_balance = dual_cam_settings["cam1"]["white_balance"]
        vx2.white_balance = dual_cam_settings["cam2"]["white_balance"]

        vx1.binning = dual_cam_settings["cam1"]["binning"]
        vx2.binning = dual_cam_settings["cam2"]["binning"]

        vx1.pad_right = dual_cam_settings["cam1"]["pad_right"]
        vx1.pad_left = dual_cam_settings["cam1"]["pad_left"]
        vx1.pad_top = dual_cam_settings["cam1"]["pad_top"]
        vx1.pad_bottom = dual_cam_settings["cam1"]["pad_bottom"]
        vx2.pad_right = dual_cam_settings["cam2"]["pad_right"]
        vx2.pad_left = dual_cam_settings["cam2"]["pad_left"]
        vx2.pad_top = dual_cam_settings["cam2"]["pad_top"]
        vx2.pad_bottom = dual_cam_settings["cam2"]["pad_bottom"]

        vx1.reverse_x = dual_cam_settings["cam1"]["reverse_x"]
        vx1.reverse_y = dual_cam_settings["cam1"]["reverse_y"]
        vx2.reverse_x = dual_cam_settings["cam2"]["reverse_x"]
        vx2.reverse_y = dual_cam_settings["cam2"]["reverse_y"]

        # Start streaming
        vx1.start()
        vx2.start()

        while True:

            # take and display pic
            if vx1._latest is None or vx2._latest is None: 
                continue

            img1 = vx1.pop(vmbpy.PixelFormat.Bgr8) 
            img2 = vx2.pop(vmbpy.PixelFormat.Bgr8)


            # # rotate image by 180 degrees
            # img1 = cv2.rotate(img1, cv2.ROTATE_180) # only necessary for current prototype

            # trim imgages to the same size
            start_trim = time.time()
            if img1.shape[0] < img2.shape[0]:
                img2 = img2[:img1.shape[0], :, :]
            elif img2.shape[0] < img1.shape[0]:
                img1 = img1[:img2.shape[0], :, :]
            if img1.shape[1] < img2.shape[1]:
                img2 = img2[:, :img1.shape[1], :]
            elif img2.shape[1] < img1.shape[1]:
                img1 = img1[:, :img2.shape[1], :]
            end_trim = time.time()
        
            start_downsamp = time.time()
            img1_downsamp = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
            img2_downsamp = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
            end_downsamp = time.time()

            left_blob = get_brightest_contour(img1_downsamp, otsu_adjustment=15)
            right_blob = get_brightest_contour(img2_downsamp, otsu_adjustment=15)

            if left_blob is None or right_blob is None:
                disp = np.concatenate((img1, img2), axis=1)
                cv2.imshow('', disp)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break
                continue

        
            left_blob_point = np.array([list(left_blob["center"])]).astype(np.float32) * 2
            right_blob_point = np.array([list(right_blob["center"])]).astype(np.float32) * 2
            left_blob_undistorted = cv2.undistortPoints(left_blob_point, stereo_params.cam1_params.K, stereo_params.cam1_params.D, P = stereo_params.cam1_params.K)
            right_blob_undistorted = cv2.undistortPoints(right_blob_point, stereo_params.cam2_params.K, stereo_params.cam2_params.D, P = stereo_params.cam2_params.K)
            left_point = left_blob_undistorted[0][0].astype(int)
            right_point = right_blob_undistorted[0][0].astype(int)
            end_undistort = time.time()
     
            uvs1 = np.array([left_point])
            uvs2 = np.array([right_point])
            # draw the blobs as markers
            if DISPLAY:
                cv2.drawMarker(img1, left_blob_point.astype(int)[0], (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
                cv2.drawMarker(img2, right_blob_point.astype(int)[0], (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
                disp = np.concatenate((img1, img2), axis=1)
                cv2.imshow('', disp)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break

except KeyboardInterrupt:
    pass
except Exception as e:
    print(traceback.format_exc())