"""
Functions for 3D localization using the stereo camera system
"""

import cv2 as cv
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

# ########################
# VISUALIZATION GLOBALS
# ########################
rgb_colors = {} 
for name, hex in colors.cnames.items():
    rgb = colors.to_rgb(hex) 
    if rgb[0]*255 + rgb[1]*255+ rgb[2]*255 > 600: # ignore very light colors
        continue
    rgb_colors[name] = rgb
colors_list = np.array(list(rgb_colors.values()))
colors_names = np.array(list(rgb_colors.keys()))
color_indices = np.arange(len(colors_list))
np.random.shuffle(color_indices)  # shuffle colors to avoid similar shades next to each other
colors_list = [(int(255*c[2]), int(255*c[1]), int(255*c[0])) for c in colors_list[color_indices]] # invert to match OpenCV's BGR format with matplotlib names
colors_names = colors_names[color_indices]

# ########################
# TRIANGULATION
# ########################
def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
            P1[0,:] - point1[0]*P1[2,:],
            point2[1]*P2[2,:] - P2[1,:],
            P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]

def triangulate(uvs1, uvs2, stereo_cam_params, display = False):

    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)
    
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = stereo_cam_params.cam1_params.K @ RT1 #projection matrix for C1
 
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([stereo_cam_params.R, stereo_cam_params.T], axis = -1)
    P2 = stereo_cam_params.cam2_params.K @ RT2 #projection matrix for C2

    # https://stackoverflow.com/questions/66361968/is-cv2-triangulatepoints-just-not-very-accurate
    # https://github.com/lambdaloop/anipose/issues/41
    # uvs1 = np.array(uvs1).reshape(-1, 1, 2)
    # uvs2 = np.array(uvs2).reshape(-1, 1, 2)
    # print(uvs1, uvs2)
    # p3ds_homog = cv2.triangulatePoints(P1, P2, uvs1, uvs2)
    # p3ds_homog = p3ds_homog.transpose()
    # print("Homogeneous 3D points: ", p3ds_homog)
    # p3ds = cv2.convertPointsFromHomogeneous(p3ds_homog).reshape(-1, 3)
    # print("3D points: ", p3ds)

    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        for i, p3d in enumerate(p3ds):
            ax.scatter(p3d[0], p3d[1], p3d[2], c = colors_names[i % len(colors_list)])
        plt.show()
    
    p3ds = np.array(p3ds)
    return p3ds

# ########################
# DEPTH FROM DISPARITY
# ########################
def depth_map_from_disparity(left_frame, right_frame, stereo_cam_params):
    # https://albertarmea.com/post/opencv-stereo-camera/
   
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            stereo_cam_params.cam1_params.K, stereo_cam_params.cam1_params.D, stereo_cam_params.cam1_rectification,
            stereo_cam_params.cam1_projection, stereo_cam_params.cam1_params.cal_dims, cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            stereo_cam_params.cam2_params.K, stereo_cam_params.cam2_params.D, stereo_cam_params.cam2_rectification,
            stereo_cam_params.cam2_projection, stereo_cam_params.cam2_params.cal_dims, cv2.CV_32FC1)

    fixedLeft = cv2.remap(left_frame, leftMapX, leftMapY,  cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,0)
    fixedRight = cv2.remap(right_frame, rightMapX, rightMapY, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,0)

    fixed_right_vis = fixedRight.copy()
    fixed_left_vis = fixedLeft.copy()

    # draw ROI rectangles
    left_roi = stereo_cam_params.cam1ROI
    right_roi = stereo_cam_params.cam2ROI
    cv2.rectangle(fixed_right_vis, (right_roi[0], right_roi[1]), (right_roi[0] + right_roi[2], right_roi[1] + right_roi[3]), (0, 255, 0), 2)
    cv2.rectangle(fixed_left_vis, (left_roi[0], left_roi[1]), (left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]), (0, 255, 0), 2)
    
    # draw horizontal lines
    for i in range(20, fixedRight.shape[0], 40):
        cv2.line(fixed_right_vis, (0, i), (fixedRight.shape[1], i), (0, 0, 255), 1)
        cv2.line(fixed_left_vis, (0, i), (fixedLeft.shape[1], i), (0, 0, 255), 1)
    display = np.hstack((fixed_left_vis, fixed_right_vis))
    cv2.imshow('Rectified images', display)
    cv2.waitKey(0)

    def nothing(x):
        pass
    
    cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp',600,600)
    
    cv2.createTrackbar('numDisparities','disp',1,17,nothing)
    cv2.createTrackbar('blockSize','disp',5,50,nothing)
    cv2.createTrackbar('preFilterType','disp',1,1,nothing)
    cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
    cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
    cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
    cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
    cv2.createTrackbar('speckleRange','disp',0,100,nothing)
    cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
    cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
    cv2.createTrackbar('minDisparity','disp',5,25,nothing)
    
    # Creating an object of StereoBM algorithm
    stereo = cv2.StereoBM_create()
    
    while True:
    
        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
        blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType','disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
        speckleRange = cv2.getTrackbarPos('speckleRange','disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
        minDisparity = cv2.getTrackbarPos('minDisparity','disp')
        
        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
    
        # Calculating disparity using the StereoBM algorithm
        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(grayLeft, grayRight)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it 
        # is essential to convert it to CV_32F and scale it down 16 times.
    
        # Converting to float32 
        disparity = disparity.astype(np.float32)
    
        # Scaling down the disparity values and normalizing them 
        disparity = (disparity/16.0 - minDisparity)/numDisparities
    
        # Displaying the disparity map
        cv2.imshow("disp",disparity)
    
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        
    

def point_depth_from_disparity(uvs1, uvs2, stereo_cam_params):
    uvs1 = np.array(uvs1).astype(np.float32)
    uvs2 = np.array(uvs2).astype(np.float32)

    uvs1_undistorted = cv2.undistortPoints(uvs1, stereo_cam_params.cam1_params.K, stereo_cam_params.cam1_params.D, R = stereo_cam_params.cam1_rectification, P = stereo_cam_params.cam1_projection)
    uvs2_undistorted = cv2.undistortPoints(uvs2, stereo_cam_params.cam2_params.K, stereo_cam_params.cam2_params.D, R = stereo_cam_params.cam2_rectification, P = stereo_cam_params.cam2_projection)
    print("uvs1:", uvs1_undistorted)
    print("uvs2:", uvs2_undistorted)
    disparities = uvs1_undistorted[:, :, 0] - uvs2_undistorted[: , :, 0]
    print("Disparities: ", disparities)

    # https://stackoverflow.com/questions/16833267/how-to-estimate-baseline-using-calibration-matrix
    c = -1 * np.linalg.inv(stereo_cam_params.R) @ stereo_cam_params.T
    baseline = np.linalg.norm(c)
    print("Basline: ", baseline)

    # https://answers.opencv2.org/question/139166/focal-length-from-calibration-parameters/
    # pixel size 3.45 µm × 3.45 µm => 0.00345 mm x 0.00345 mm (https://www.alliedvision.com/en/products/alvium-configurator/alvium-1800-u/240/)
    pixel_size = 0.00345 #pixel size in world unit (mm)
    fx = stereo_cam_params.cam1_projection[0,0] #cam1_projection == cam2_projection
    focal_length = fx * pixel_size #convert focal length in pixels to mm
    print("Focal length: ", focal_length)
    
    # https://answers.opencv2.org/question/199915/how-to-get-depth-from-disparity-map/#:~:text=Z%20%3D%20B*f%20%2F%20disparity%20is%20the%20correct%20formula%20to,get%20the%20depth%20in%20mm.&text=the%20resulting%20Camera%20matrix%20holds%20the%20value(s)%20for%20f%20.
    depth = focal_length * baseline / disparities
    # [TODO] check if depth is accurate
    # [TODO] get x and y 


    return depth

# ########################
# TESTING
# ########################
def visualize_box_results(bottom_points_path, top_points_path):
    """
    Ground truth box dimensions:
    l = 21.2 cm
    h = 14.5 cm
    w = 39.5 cm
    """
    bottom_points = np.load(bottom_points_path)
    top_points = np.load(top_points_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim3d(0, 1000)
    # ax.set_ylim3d(0, 1000)
    # ax.set_zlim3d(0, 1000)

    for i, p3d in enumerate(bottom_points):
        ax.scatter(p3d[0], p3d[1], p3d[2], c = 'r')
    for i, p3d in enumerate(top_points):
        ax.scatter(p3d[0], p3d[1], p3d[2], c = 'b')
    ax.plot(xs = [bottom_points[0][0], top_points[0][0]], ys = [bottom_points[0][1], top_points[0][1]], zs = [bottom_points[0][2], top_points[0][2]], c = 'grey')
    height = np.linalg.norm(bottom_points[0] - top_points[0])
    mid_point = (bottom_points[0] + top_points[0]) / 2
    ax.text(mid_point[0], mid_point[1], mid_point[2], f'h = {height:.2f} mm', color='black')

    ax.plot(xs = [bottom_points[0][0], bottom_points[1][0]], ys = [bottom_points[0][1], bottom_points[1][1]], zs = [bottom_points[0][2], bottom_points[1][2]], c = 'grey')
    width = np.linalg.norm(bottom_points[0] - bottom_points[1])
    mid_point = (bottom_points[0] + bottom_points[1]) / 2
    ax.text(mid_point[0], mid_point[1], mid_point[2], f'w = {width:.2f} mm', color='black')

    # width2 = np.linalg.norm(bottom_points[0] - bottom_points[1])
    # print("Width2: ", width2)

    ax.plot(xs = [top_points[0][0], top_points[2][0]], ys = [top_points[0][1], top_points[2][1]], zs = [top_points[0][2], top_points[2][2]], c = 'grey')
    length = np.linalg.norm(top_points[0] - top_points[2])
    mid_point = (top_points[0] + top_points[2]) / 2
    ax.text(mid_point[0], mid_point[1], mid_point[2], f'l = {length:.2f} mm', color='black')

    # ax.plot(xs = [top_points[2][0], top_points[3][0]], ys = [top_points[3][1], top_points[3][1]], zs = [top_points[2][2], top_points[3][2]], c = 'grey')
    # ax.plot(xs = [top_points[3][0], top_points[1][0]], ys = [top_points[3][1], top_points[1][1]], zs = [top_points[3][2], top_points[1][2]], c = 'grey')
    # ax.plot(xs = [bottom_points[2][0], bottom_points[3][0]], ys = [bottom_points[3][1], bottom_points[3][1]], zs = [bottom_points[2][2], bottom_points[3][2]], c = 'grey')
    # ax.plot(xs = [bottom_points[3][0], bottom_points[1][0]], ys = [bottom_points[3][1], bottom_points[1][1]], zs = [bottom_points[3][2], bottom_points[1][2]], c = 'grey')
    
    
    plt.show()

