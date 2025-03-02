from stereo_calibration import StereoCameraCalibrator, undistort
from localization import triangulate, colors_list
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import lib.vxcam as vxcam
import vmbpy
import time
import os
import pickle


SPLIT_AT = 885 # pixel column at which to split the dual-view image, will be updated to 0.5*width within vimba_cap

def single_vimba_cap(frame_folder_path = None, video_output_path = None, manual = False):
    if frame_folder_path is not None:
        os.makedirs(f"{frame_folder_path}/LEFT", exist_ok = True)
        os.makedirs(f"{frame_folder_path}/RIGHT", exist_ok = True)

    i = 0
    m = 0
    N = 100
    fps = np.zeros(N)
    t0 = time.time()
    with vxcam.VXCam() as vx:
        vx.pixel_format = vmbpy.PixelFormat.Bgr8
        vx.auto_exposure = 'Off'
        vx.exposure_us = 15000
        vx.gain_db = 8
        vx.white_balance = 'Once'
        
        # Padding is from [0..1] relative to the left, right, top, bottom extents
        vx.binning = 1
        vx.pad_right = 0.01
        vx.pad_left = 0.06
        vx.pad_top = 0.01
        vx.pad_bottom = 0.16

        if video_output_path is not None:
            cap = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (vx.width, vx.height))

        # Start streaming, can't update again until vx.stop() is called
        vx.start()
        try:
            while True: 
                # Pop latest image and convert to openCV format
                img = vx.pop(vmbpy.PixelFormat.Bgr8) 
                if img is None: continue
                
                img1 = img[:, :SPLIT_AT]
                img2 = img[:, SPLIT_AT:]
                # trim imgages to the same size
                if img1.shape[0] < img2.shape[0]:
                    img2 = img2[:img1.shape[0], :, :]
                elif img2.shape[0] < img1.shape[0]:
                    img1 = img1[:img2.shape[0], :, :]
                if img1.shape[1] < img2.shape[1]:
                    img2 = img2[:, :img1.shape[1], :]
                elif img2.shape[1] < img1.shape[1]:
                    img1 = img1[:, :img2.shape[1], :]
                

                # reflect img1 over the vertical axis
                img1 = cv2.flip(img1, 1)
                img1 = cv2.flip(img1, 0)

                # vis_img = img.copy()
                vis_img = np.hstack((img1, img2))
                cv2.putText(vis_img, f'{np.mean(fps):.0f}fps', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, thickness=2)
                cv2.line(vis_img, (SPLIT_AT, 0), (SPLIT_AT, vx.height), 255, thickness=2)
                cv2.imshow('', vis_img)
                k = cv2.waitKey(1)
                if k == ord('q'): break
                elif k == ord('a') and frame_folder_path is not None and manual:
                    cv2.imwrite(f"{frame_folder_path}/LEFT/{m}.png", img1)
                    cv2.imwrite(f"{frame_folder_path}/RIGHT/{m}.png", img2)
                    m += 1
                    continue
                
                if not manual:
                    if video_output_path is not None:
                        cap.write(vis_img)
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
            if video_output_path is not None:
                cap.release()
    
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def split_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    left_out = cv2.VideoWriter(output_path.replace(".mp4", "_left.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width//2, height))
    right_out = cv2.VideoWriter(output_path.replace(".mp4", "_right.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width//2, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        left = frame[:, :SPLIT_AT]
        right = frame[:, SPLIT_AT:]
        left_out.write(left)
        right_out.write(right)
    cap.release()
    left_out.release()
    right_out.release()



def select_test_correspondences(input_path, stereo_cam_params, undistort_frames = True):

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
    extension = os.path.splitext(input_path)[-1]
    if extension in [".mp4", ".avi", ".mov"]:
        cap = cv2.VideoCapture(input_path)
        while True:
            ret, img = cap.read()
            if not ret:
                break
            left = img[:, :SPLIT_AT]
            right = img[:, SPLIT_AT:]
            if undistort_frames:
                left = undistort(left, stereo_cam_params.cam1_params)
                right = undistort(right, stereo_cam_params.cam2_params)
            vis_img = np.hstack((left.copy(), right.copy()))
            vis_img = cv2.line(vis_img, (left.shape[1], 0), (left.shape[1], vis_img.shape[0]), (255, 255, 0), 2)
            cv2.imshow("Select test frame ([q] to select and exit)", vis_img)
            if cv2.waitKey(1) == ord('q'): break
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(input_path)
        left = img[:, :SPLIT_AT]
        right = img[:, SPLIT_AT:]
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


def test_undistort(video_path, camParams, left = True):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if left:
            frame = frame[:, :SPLIT_AT]
        else:
            frame = frame[:, SPLIT_AT:]
        undistorted_img = undistort(frame, camParams)
        display = np.hstack((frame, undistorted_img))
        cv2.imshow('Original + Undistorted', display)
        if cv2.waitKey(1) == ord('q'): break
    cv2.destroyAllWindows()
 
def keypoint_testing(video_path, stereo_cam_params, d=.99):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, img = cap.read()
        if not ret:
            print("Can't open video")
            return
        else:
            break
        
    left = img[:, :SPLIT_AT]
    right = img[:, SPLIT_AT:]
    left = undistort(left, stereo_cam_params.cam1_params)
    right = undistort(right, stereo_cam_params.cam2_params)

    # get keypoint matches
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(left, None)
    kp2, des2 = sift.detectAndCompute(right, None)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except: # prevent later errors from occuring if insufficient or garbage matches are found
        return None, None, float('-inf'), float('inf'), float('-inf'), float('inf')
    good = [m for (m, n) in matches if m.distance < d * n.distance]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC) # Compute a perspective transformation between the points just to get the mask
    
    left_vis = left.copy()
    right_vis = right.copy()
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(255, 0, 255), singlePointColor=(255, 255, 255), matchesMask=matchesMask)
    lrm = cv2.drawMatches(left_vis, kp1, right_vis, kp2, good, None, **draw_params)
    lrm = cv2.cvtColor(lrm, cv2.COLOR_BGR2RGB)
    plt.imshow(lrm)
    plt.show()

    print(len(good), len(matchesMask))
    uvs1 = [np.array(list(kp1[m.queryIdx].pt)).astype(int) for i, m in enumerate(good) if matchesMask[i] == 1]
    uvs2 = [np.array(list(kp2[m.trainIdx].pt)).astype(int) for i, m in enumerate(good) if matchesMask[i] == 1]
    triangulate(uvs1, uvs2, left, right, stereo_cam_params)

