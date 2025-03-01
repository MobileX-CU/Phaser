"""
Functions for ata collection from the stereo camera + laser system, 
supporting stereo-camera calibration, laesr pose recovery, and laser mapping.
"""

import lib.mirror as mirror
import lib.vxcam as vxcam
import vmbpy
from camera_settings import dual_cam_settings

import sys
import numpy as np
import traceback
import cv2
import time
import os
import math
import matplotlib.pyplot as plt

def generate_spiral(step = 8, loops1 = 6, loops2 = 5, spiral_scale = 1):
    a = 0
    b = 2
    theta = 0
    r = a

    # initial spiral points
    xs = []
    ys = []
    while theta < 2 * loops1 * math.pi:
        loop_num = theta / (2 * math.pi)
        if r == 0:
            theta += step
        else:
            theta += step / r
        r = a + b*theta
        r_eff = r
        x = int(r_eff*math.cos(theta))
        y = int(r_eff*math.sin(theta))
        xs.append(x)
        ys.append(y)

    # start drawing a new, more compressed spiral at last radius covered by first spiral,
    # for improved resolution at the edges of a wide angle lens's field of view
    last_r  = r
    theta = 0
    while theta < 2 * loops2 * math.pi:
        loop_num = theta / (2 * math.pi)
        theta += step / r
        r = last_r + b*theta
        if int(loop_num) == 0:
            r_eff = r
        else:
            r_eff = r - ((loop_num - 1) * 8)
        x = int(r_eff*math.cos(theta))
        y = int(r_eff*math.sin(theta))
        xs.append(x)
        ys.append(y)
    x_scale = max(xs) 
    y_scale = max(ys) 
    xs = np.array(xs) / x_scale
    ys = np.array(ys) / y_scale
    xs = np.around(xs, 2) * spiral_scale
    ys = np.around(ys, 2) * spiral_scale
    return xs, ys

def mapping_collection(image_output_path, manually_approve = False, mirror_range_scale = 1, spiral_scale = 1):

    scan_xs, scan_ys = generate_spiral(spiral_scale = spiral_scale)
    scan_points = list(zip(scan_xs, scan_ys))

    # plot the spiral
    plt.scatter(scan_xs, scan_ys)
    plt.xlabel("Mir x")
    plt.ylabel("Mir y")
    plt.axis("equal")
    plt.show()
    
    # set up mirror
  
    sn = 'BPAA1034'
    loc = 'lpd_v3'
    mir = mirror.SerialMirror(sn, loc, '/dev/cu.usbmodem00000000001A1', range_scale=mirror_range_scale, mirrors_csv_path='mirrors.csv')
    mir.enable()

    os.makedirs(f"{image_output_path}/LEFT", exist_ok=True)
    os.makedirs(f"{image_output_path}/RIGHT", exist_ok=True)

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
            vx1.exposure_us = dual_cam_settings["cam1"]["mapping_exposure_us"]
            vx2.exposure_us =  dual_cam_settings["cam2"]["mapping_exposure_us"]
            vx1.gain_db =  dual_cam_settings["cam1"]["mapping_gain_db"]
            vx2.gain_db = dual_cam_settings["cam2"]["mapping_gain_db"]
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

            # scan here
            scan_num = 0
            for i, s in enumerate(scan_points):
                print(f"Scan {i}/{len(scan_points)}")
                mir.steer(s, expect='OK')
                        
                # take and display pic
                time.sleep(0.3)
                while True: 
                    if vx1._latest is None or vx2._latest is None: 
                        continue
                    else:
                        break

                img1 = vx1.pop(vmbpy.PixelFormat.Bgr8) 
                img2 = vx2.pop(vmbpy.PixelFormat.Bgr8)

                # rotate image by 180 degrees
                # img1 = cv2.rotate(img1, cv2.ROTATE_180) # only necessary for current prototype

                # trim imgages to the same size
                if img1.shape[0] < img2.shape[0]:
                    img2 = img2[:img1.shape[0], :, :]
                elif img2.shape[0] < img1.shape[0]:
                    img1 = img1[:img2.shape[0], :, :]
                if img1.shape[1] < img2.shape[1]:
                    img2 = img2[:, :img1.shape[1], :]
                elif img2.shape[1] < img1.shape[1]:
                    img1 = img1[:, :img2.shape[1], :]

                disp = np.concatenate((img1, img2), axis=1)
                cv2.imshow('', disp)
                
                if manually_approve:
                    k = cv2.waitKey(0)
                    if k == ord('a'):
                        print("Saving picture")
                        cv2.imwrite(f"{image_output_path}/LEFT/{scan_num}_x{x_steer}_y{y_steer}.png", img1)
                        cv2.imwrite(f"{image_output_path}/RIGHT/{scan_num}_x{x_steer}_y{y_steer}.png", img2)
                        scan_num += 1
                else:
                    k = cv2.waitKey(1)
                    if k == ord('q'):
                        raise KeyboardInterrupt
                    x_steer = s[0]
                    y_steer = s[1]
                    cv2.imwrite(f"{image_output_path}/LEFT/{scan_num}_x{x_steer}_y{y_steer}.png", img1)
                    cv2.imwrite(f"{image_output_path}/RIGHT/{scan_num}_x{x_steer}_y{y_steer}.png", img2)
                    scan_num += 1
               
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(traceback.format_exc())

    # Cleanup
    mir.disable()

def pose_recovery_collection(image_output_path, num_planes, mirror_range_scale = 0.5):

    # scan laser in only x direction
    axis_range = np.arange(-1, 1, .02)
    
    # set up mirror
    sn = 'BPAA1034'
    loc = 'lpd_v3'
    mir = mirror.SerialMirror(sn, loc, '/dev/cu.usbmodem00000000001A1', range_scale=mirror_range_scale, mirrors_csv_path='mirrors.csv')
    mir.enable()

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
            vx1.exposure_us = dual_cam_settings["cam1"]["scanning_exposure_us"]
            vx2.exposure_us = dual_cam_settings["cam2"]["scanning_exposure_us"]
            vx1.gain_db =  dual_cam_settings["cam1"]["scanning_gain_db"]
            vx2.gain_db = dual_cam_settings["cam2"]["scanning_gain_db"]
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

            # Start scanning
            for plane in range(num_planes):
                print(f"Ready to scan depth number {plane}")
                for ax in range(2):
                    if ax == 0:
                        axis = "x"
                    else:
                        axis = "y"
                    
                    os.makedirs(f"{image_output_path}/plane_{plane}/{axis}/LEFT", exist_ok=True)
                    os.makedirs(f"{image_output_path}/plane_{plane}/{axis}/RIGHT", exist_ok=True)
                    
                    scan_num = 0
                    for i in axis_range:
                        if ax == 0:
                            x_steer = i
                            y_steer = 0
                        else:
                            x_steer = 0
                            y_steer = i

                        mir.steer((x_steer, y_steer), expect='OK')
                        
                        # take and display pic
                        time.sleep(0.1)
                        while True: 
                            if vx1._latest is None or vx2._latest is None: 
                                continue
                            else:
                                break

                        img1 = vx1.pop(vmbpy.PixelFormat.Bgr8) 
                        img2 = vx2.pop(vmbpy.PixelFormat.Bgr8)

                        # rotate image by 180 degrees
                        # img1 = cv2.rotate(img1, cv2.ROTATE_180) # only necessary for current prototype

                        # trim imgages to the same size
                        if img1.shape[0] < img2.shape[0]:
                            img2 = img2[:img1.shape[0], :, :]
                        elif img2.shape[0] < img1.shape[0]:
                            img1 = img1[:img2.shape[0], :, :]
                        if img1.shape[1] < img2.shape[1]:
                            img2 = img2[:, :img1.shape[1], :]
                        elif img2.shape[1] < img1.shape[1]:
                            img1 = img1[:, :img2.shape[1], :]

                        disp = np.concatenate((img1, img2), axis=1)
                        cv2.imshow("Press a to take pic and move to next scan, n to move to next axis, q to quit whole program, any other key to just move the laser", disp)
                        k = cv2.waitKey(0)
                        if k == ord("a"):
                    
                        # inp = input()
                        # if inp == "a":
                        #     print("Saving picture")
                            
                            cv2.imwrite(f"{image_output_path}/plane_{plane}/{axis}/LEFT/{scan_num}_x{x_steer:.2f}_y{y_steer:.2f}.png", img1)
                            cv2.imwrite(f"{image_output_path}/plane_{plane}/{axis}/RIGHT/{scan_num}_x{x_steer:.2f}_y{y_steer:.2f}.png", img2)

                            scan_num += 1
                        elif k == ord("n"):
                            break
                        elif k == ord("q"):
                            raise KeyboardInterrupt

                next_plane = input("Press enter to move to next depth.")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(traceback.format_exc())

    # Cleanup
    mir.disable()

def dual_vimba_cap(frame_folder_path = None, video_output_name = None, manual = False):

    with vmbpy.VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        num_cams = len(cams)
        print(f"Found {num_cams} cameras.")
        if num_cams != 2:
            print("Need min 2 cameras to run this function, exiting.")
            return

    if frame_folder_path is not None:
        os.makedirs(f"{frame_folder_path}/LEFT", exist_ok = True)
        os.makedirs(f"{frame_folder_path}/RIGHT", exist_ok = True)
    
    i = 0
    m = 0
    N = 100
    fps = np.zeros(N)
    t0 = time.time()
    with vxcam.VXCam() as vx1, vxcam.VXCam(1) as vx2:
        
        vx1.pixel_format = dual_cam_settings["cam1"]["pixel_format"]
        vx2.pixel_format = dual_cam_settings["cam2"]["pixel_format"]
        vx1.auto_exposure = dual_cam_settings["cam1"]["auto_exposure"]
        vx2.auto_exposure = dual_cam_settings["cam2"]["auto_exposure"]
        vx1.exposure_us = dual_cam_settings["cam1"]["checkerboard_exposure_us"]
        vx2.exposure_us = dual_cam_settings["cam2"]["checkerboard_exposure_us"]
        vx1.gain_db = dual_cam_settings["cam1"]["checkerboard_gain_db"]
        vx2.gain_db = dual_cam_settings["cam2"]["checkerboard_gain_db"]
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

        if video_output_name is not None:
            cap1 = cv2.VideoWriter(video_output_name + "_left.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 60, (min(vx1.width, vx2.width), min(vx1.height, vx2.height)))
            cap2 = cv2.VideoWriter(video_output_name + "_right.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 60, (min(vx1.width, vx2.width), min(vx1.height, vx2.height)))
     
        # Start streaming, can't update again until vx.stop() is called
        vx1.start()
        vx2.start()

        try:
            while True: 
                # Pop latest image and convert to openCV format
                if vx1._latest is None or vx2._latest is None: 
                    continue

                img1 = vx1.pop(vmbpy.PixelFormat.Bgr8) 
                img2 = vx2.pop(vmbpy.PixelFormat.Bgr8)

                # rotate image by 180 degrees
                # img1 = cv2.rotate(img1, cv2.ROTATE_180) # only necessary for current prototype

                # trim imgages to the same size
                if img1.shape[0] < img2.shape[0]:
                    img2 = img2[:img1.shape[0], :, :]
                elif img2.shape[0] < img1.shape[0]:
                    img1 = img1[:img2.shape[0], :, :]
                if img1.shape[1] < img2.shape[1]:
                    img2 = img2[:, :img1.shape[1], :]
                elif img2.shape[1] < img1.shape[1]:
                    img1 = img1[:, :img2.shape[1], :]

                vis_img = np.hstack((img1, img2))
                cv2.putText(vis_img, f'{np.mean(fps):.0f}fps', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, thickness=2)
                cv2.imshow('Press a to save this frame, q to quit, anything else to capture another frame.', vis_img)
                k = cv2.waitKey(1)
                if k == ord('q'): break
                elif k == ord('a') and frame_folder_path is not None and manual:
                    cv2.imwrite(f"{frame_folder_path}/LEFT/{m}.png", img1)
                    cv2.imwrite(f"{frame_folder_path}/RIGHT/{m}.png", img2)
                    m += 1
                    continue
                
                if not manual:
                    if video_output_name is not None:
                        cap1.write(img1)
                        cap2.write(img2)
                    if frame_folder_path is not None :
                        cv2.imwrite(f"{frame_folder_path}/LEFT/{i}.png", img1)
                        cv2.imwrite(f"{frame_folder_path}/RIGHT/{i}.png", img2)
                
                # FPS calculations. Won't be accurate if in manual mode.
                fps[i % N] = 1/(time.time() - t0)
                t0 = time.time()
                i += 1
        except KeyboardInterrupt:
            pass
        finally:
            pass
            if video_output_name is not None:
                cap1.release()
                cap2.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)


# dual_vimba_cap(frame_folder_path = "stereo_calibration_input/jul21/images", manual = True)
# pose_recovery_collection("pose_recovery_input/jul21/images", 4, mirror_range_scale = 1)
# mapping_collection("mapping_input/jul21/images", manually_approve = False, mirror_range_scale=1, spiral_scale = .27)