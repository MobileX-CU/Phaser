"""
Mobile target tracking and steering of the laser after calibratin.
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


"""
Deployment workflow, per iteration in continuous loop: 
1) Detect a point in both views to triangulate to.
2) Get 3D coord of point wrt cameras using triangulation.
3) Transform the 3D coordinate to the laser's reference frame
4) Get the angles in both dimensions of the target relative to the laser origin. 
5) Query the steering mappings with the angles to obtain the steering command
6) Send the steering command to the mirror and watch it (hopefully) steer to the target.
"""
 
# set the variables appropriately below
DISPLAY = True # display the camera views and blob detection. slows down steering a bit, but helpful for developemt and demos
stereo_params_path = "params/example_stereo_params.pkl"
laser_params_path = "params/example_laser_params.pkl"
mirror_x_params_path = "params/example_mir_x_popt.npy"
mirror_y_params_path = "params/example_mir_y_popt.npy"

# load the variables
with open(stereo_params_path, "rb") as f:
    stereo_params = pickle.load(f)

with open(laser_params_path, "rb") as f:
    laser_params = pickle.load(f)
mir_x_popt = np.load(mirror_x_params_path)
mir_y_popt = np.load(mirror_y_params_path)



def func_degree2(theta, phi, p00, p10, p01, p20, p11, p02):
    """
    Polynomial function of degree 2, defined by the coefficients p*
    
    Parameters:
    data : 2xN array 
        Array of points, where each column is a point in 2D space
    p00, p10, p01, p20, p11, p02: ints
        Coefficients of the polynomial function

    Returns:
    z : 1xN array 
        Array of z values for each point, computed by the polynomial function

    """
    return p00 + p10*theta+ p01*phi + p20*theta**2 + p11*theta*phi + p02*phi**2 


# set up mirror
scale = 1 #IMPORTANT: make sure the scale is the same as the one used for mapping
sn = 'BPAA1034'
loc = 'lpd_v3'
mir = mirror.SerialMirror(sn, loc, '/dev/cu.usbmodem00000000001A1', range_scale=scale, mirrors_csv_path='mirrors.csv')
mir.enable()

cap_times = []
trim_times = []
undistort_times = []
blob_times = []
triangulate_times = []
transform_times = []
mapping_times = []
steer_times = []
downsaple_times = []

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
            start_cap = time.time()
            if vx1._latest is None or vx2._latest is None: 
                continue

            img1 = vx1.pop(vmbpy.PixelFormat.Bgr8) 
            img2 = vx2.pop(vmbpy.PixelFormat.Bgr8)
            end_cap = time.time()
            cap_times.append(end_cap - start_cap)

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
            trim_times.append(end_trim - start_trim)
        
            # detect blob in each image
            # below for blue LED detection:
            # start_undistort = time.time()
            # img1 = undistort(img1.copy(), stereo_params.cam1_params)
            # img2 = undistort(img2.copy(), stereo_params.cam2_params)
            # end_undistort = time.time()
            # undistort_times.append(end_undistort - start_undistort)

            start_downsamp = time.time()
            img1_downsamp = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
            img2_downsamp = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
            end_downsamp = time.time()
            downsaple_times.append(end_downsamp - start_downsamp)

            start_blob = time.time()
            left_blob = get_brightest_contour(img1_downsamp, otsu_adjustment=5)
            right_blob = get_brightest_contour(img2_downsamp, otsu_adjustment=5)
            end_blob = time.time()
            blob_times.append(end_blob - start_blob)
            if left_blob is None or right_blob is None:
                print("No blob detected in one of the images, try again.")
                if DISPLAY:
                    disp = np.concatenate((img1, img2), axis=1)
                    cv2.imshow('', disp)
                    if cv2.waitKey(1) == ord('q'):
                        cv2.destroyAllWindows()
                        break
                continue
    
            start_undistort = time.time()
            left_blob_point = np.array([list(left_blob["center"])]).astype(np.float32) * 2
            right_blob_point = np.array([list(right_blob["center"])]).astype(np.float32) * 2
            left_blob_undistorted = cv2.undistortPoints(left_blob_point, stereo_params.cam1_params.K, stereo_params.cam1_params.D, P = stereo_params.cam1_params.K)
            right_blob_undistorted = cv2.undistortPoints(right_blob_point, stereo_params.cam2_params.K, stereo_params.cam2_params.D, P = stereo_params.cam2_params.K)
            left_point = left_blob_undistorted[0][0].astype(int)
            right_point = right_blob_undistorted[0][0].astype(int)
            end_undistort = time.time()
            undistort_times.append(end_undistort - start_undistort)

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
            
            # for manual operation tests: point in the views to triangulate to
            # save them both for selection purposes
            # cv2.imwrite("img1.png", img1)
            # cv2.imwrite("img2.png", img2)
            # uvs1, uvs2, _, _ = select_test_correspondences("img1.png", "img2.png", stereo_params)

            # triangulate and transform
            start_triangulate = time.time()
            p3ds = triangulate(uvs1, uvs2, stereo_params)
            end_triangulate = time.time()
            triangulate_times.append(end_triangulate - start_triangulate)

            # laser_B = np.hstack((laser_rotation[0].reshape((3, 1)), laser_rotation[1].reshape((3, 1)), laser_rotation[2].reshape((3, 1))))
            # p3ds_transformed = p3ds.copy()
            # p3ds_transformed[:, 0] = p3ds_transformed[:, 0] - laser_translation[0]
            # p3ds_transformed[:, 1] = p3ds_transformed[:, 1] - laser_translation[1]
            # p3ds_transformed[:, 2] = p3ds_transformed[:, 2] - laser_translation[2]
            # p3ds_transformed = np.matmul(laser_B.T, p3ds_transformed.T).T
            start_transform = time.time()
            p3ds_transformed = transform_cam2laser(p3ds, laser_params)
            end_transform = time.time()
            transform_times.append(end_transform - start_transform)
            print("3D point in camera frame: ", p3ds)
            print("3D point in laser frame: ", p3ds_transformed)

            start_mapping = time.time()
            target = p3ds_transformed[0]

            theta = np.rad2deg(np.arctan2(target[0], target[2]))
            phi = np.rad2deg(np.arctan2(target[1], target[2]))


            mir_x = func_degree2(theta, phi, *mir_x_popt)
            mir_y = func_degree2(theta, phi, *mir_y_popt)
            end_mapping = time.time()
            mapping_times.append(end_mapping - start_mapping)
            
            print("Steering mirror to: ", mir_x, mir_y)

            start_steer = time.time()
            mir.steer((mir_x, mir_y), expect='OK')
            end_steer = time.time()
            steer_times.append(end_steer - start_steer)

            # inp = input("press any key when ready to take next pic")
            # time.sleep(1)

        
except KeyboardInterrupt:
    pass
except Exception as e:
    print(traceback.format_exc())

# Cleanup
mir.disable()
print("Average time for capture: ", np.mean(cap_times))
print("Average time for trim: ", np.mean(trim_times))
print("Average time for undistort: ", np.mean(undistort_times))
print("Average time for blob detection: ", np.mean(blob_times))
print("Average time for triangulate: ", np.mean(triangulate_times))
print("Average time for transform: ", np.mean(transform_times))
print("Average time for mapping: ", np.mean(mapping_times))
print("Average time for steering: ", np.mean(steer_times))
print("Average time for downsampling: ", np.mean(downsaple_times))