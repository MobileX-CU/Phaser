"""
All functions and classes for calibrating the laser steering unit in the triangulation-based
tracking + steering setup, including recovering the laser's pose and basis w.r.t a stereo-camera 
device, and mapping steering device input commands to outgoing laser angles.
"""

from localization import triangulate
from stereo_calibration import *
from geometry import *
from im_proc import *
from laser_calibration import *

import matplotlib.pyplot as plt
import pickle
import cv2
import numpy as np
import glob
from skspatial.objects import Line, Vector, Point

class LaserPoseParams():
    def __init__(self, laser_pose_R, laser_pose_T = [0, 0, 0]):
        """
        Laser's basis (rotation matrix) and translation in the reference camera's coordinate system. 
        Note this is NOT the same as the R and T that constitute the extrinsic matrix of the laser, 
        which counterintuitively express the camera's position in the laser's reference frame (https://ksimek.github.io/2012/08/22/extrinsic/)
        """
        self.pose_R = laser_pose_R
        self.pose_T = np.array(laser_pose_T)
       

def transform_cam2laser(points3d, laser_pose_params):
    """
    Given laser pose, get the transform to actually apply to points in the camera's
    reference frame to get them in the laser's reference frame (https://ksimek.github.io/2012/08/22/extrinsic/)
    and apply to points
    """
    transform_init = np.hstack((laser_pose_params.pose_R, laser_pose_params.pose_T.reshape((3, 1))))
    transform_init = np.vstack((transform_init, np.array([0, 0, 0, 1])))
    transform = np.linalg.inv(transform_init)
    points3d_h = np.hstack((points3d , np.ones((points3d.shape[0], 1)))) # express 3d points as homogenous coordinates
    points3d_transformed_h = np.matmul(transform, points3d_h.T).T
    points3d_transformed =  points3d_transformed_h[:, :3]
    return points3d_transformed


###############
# POSE RECOVERY
###############
def get_laser_pose_recovery_points2d(images_path, stereo_params_path, num_planes, manual_approve = False, manual_selection = False, output_folder_path = "."):
    """
    Get the laser spots from images taken during x and y axis scans.
    
    Parameters:
    images_path : str
        Path to folder containing stereocam images of scans on multiple planes, where the internal folder structure is assumed to be: 
        images_path/
            plane_{plane_num}/
                x/
                    LEFT/
                        *.png
                    RIGHT/
                        *.png
                y/
                    LEFT/
                        *.png
                    RIGHT/
                        *.png
    stereo_params_path : str
        Path to pickle file containing stereo camera calibration parameters.
        This is only needed so that the images can be undistorted.
    num_planes : int
        Number of planes scans were taken at
    manual_approve : bool
        If True, will require user input to approve the detected laser spots in each image. 
        If False, will automatically select the brightest spot in each image.
    manual_selection : bool
        If True, will require user input to click on the laser spot in each image. 
    output_folder_path : str
        Path to folder to save .npy files containing the detected laser spots.
    
    Returns:
    None. Data is saved in npy files.
    """

    if manual_approve and manual_selection:
        raise ValueError("Cannot have both manual_approve and manual_selection set to True")
        
    with open(stereo_params_path, "rb") as f:
        stereo_cam_params = pickle.load(f)
    
    image_paths = glob.glob(f"{images_path}/plane_0/x/LEFT/*.png")
    angles = [i_path.split("/")[-1].split("_")[1].split("_")[0].split("x")[1] for i_path in image_paths]
    target_angle = angles[len(angles) // 3]
    # ensure target angle is not 0. if it is, we can't apply the similar triangles logic
    if float(target_angle) == 0:
        target_angle = angles[3 * len(angles) // 4]
    print("Target angle:", target_angle)

    # unlike uvs in the loop, these are for the target angle at all planes, rather than being separated by plane
    target_angle_uvs1 = []
    target_angle_uvs2 = []
    for plane_num in range(num_planes):
        for ax_num in range(2):
            if ax_num == 0:
                axis = "x"
            elif ax_num == 1:
                axis = "y"
            imgs_left = glob.glob(f"{images_path}/plane_{plane_num}/{axis}/LEFT/*.png")
            imgs_right = glob.glob(f"{images_path}/plane_{plane_num}/{axis}/RIGHT/*.png")
            imgs_left.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]))
            imgs_right.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]))


            uvs1 = []
            uvs2 = []
            for i in range(len(imgs_left)):
                
                if manual_selection:
                    left_blob_selections, right_blob_selections, left_frame, right_frame = select_test_correspondences(imgs_left[i], imgs_right[i], stereo_cam_params, undistort_frames=True)
                    if len(left_blob_selections) == 0 or len(right_blob_selections) == 0:
                        print("No blobs selected")
                        continue
                    
                    uvs1.append(left_blob_selections[0])
                    uvs2.append(right_blob_selections[0])
                else:
                    # print(imgs_left[i], imgs_right[i])

                    left_img = cv2.imread(imgs_left[i])
                    right_img = cv2.imread(imgs_right[i])

                    # undistort the images
                    left_img = undistort(left_img, stereo_cam_params.cam1_params)
                    right_img = undistort(right_img, stereo_cam_params.cam2_params)

                    # detect blob in each
                    left_blob_obj = get_brightest_contour(left_img, otsu_adjustment=10, contour_area_threshold=1000000)
                    left_img_vis = left_img.copy()
                    if left_blob_obj is not None:
                        # draw blob on left img
                        cv2.circle(left_img_vis, left_blob_obj["center"], left_blob_obj["radius"], (0, 255, 0), 2)
                    else:
                        print("No blob detected in left frame")    
                    
                    right_blob_obj = get_brightest_contour(right_img, otsu_adjustment=10, contour_area_threshold=1000000)
                    right_img_vis = right_img.copy()
                    if right_blob_obj is not None:
                        # draw blob on right img
                        cv2.circle(right_img_vis, right_blob_obj["center"], right_blob_obj["radius"], (0, 255, 0), 2)
                    else:
                        print("No blob detected in right frame")
                    
                
                    disp = np.hstack((left_img_vis, right_img_vis))
                    if manual_approve:
                        caption = "Press a to approve, q to quit"
                    else:
                        caption = ""
                    cv2.imshow(caption, disp)
                   
                    if manual_approve:
                        k = cv2.waitKey(0)
                        # wait for user decision to add or not
                        if k != ord('a'):
                            continue
                        elif k == ord('q'):
                            print("Destroy")
                            cv2.destroyAllWindows()
                            for i in range (1,5):
                                cv2.destroyAllWindows()
                            return 
                        else:
                            if left_blob_obj is None or right_blob_obj is None:
                                continue
                    else:
                        k = cv2.waitKey(1)
                        if k == ord('q'):
                            cv2.destroyAllWindows()
                            for i in range (1,5):
                                cv2.destroyAllWindows()
                            return
                        # automatically add if blob detected in both images, otherwise continue
                        if left_blob_obj is None or right_blob_obj is None:
                            continue
                            
                    uvs1.append(left_blob_obj["center"])
                    uvs2.append(right_blob_obj["center"])
                    # get the x angle from the img name and add to target_angle_uvs if it matches the target angle
                    x_steer = imgs_left[i].split("/")[-1].split("_")[1].split("_")[0].split("x")[1]
                    if x_steer == target_angle and axis == "x":   
                        print("Adding ", x_steer, target_angle)
                        target_angle_uvs1.append(left_blob_obj["center"])
                        target_angle_uvs2.append(right_blob_obj["center"])

            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            np.save(f"{output_folder_path}/plane_{plane_num}_{axis}_uvs1.npy", uvs1)
            np.save(f"{output_folder_path}/plane_{plane_num}_{axis}_uvs2.npy", uvs2)
            np.save(f"{output_folder_path}/target_angle_uvs1.npy", target_angle_uvs1)
            np.save(f"{output_folder_path}/target_angle_uvs2.npy", target_angle_uvs2)
            print(f"Saved points for plane {plane_num} axis {axis} at {output_folder_path}/plane_{plane_num}_{axis}")

    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)

def get_laser_pose_recovery_points3d(stereo_cam_params, num_planes, input_folder_path = ".", output_folder_path = None, skip_planes = []):
    """
    Using 2D points from stereopair detected and saved in get_laser_pose_recovery_points2d, triangulate them to get corresponding 3D points. 

    Parameters:
    stereo_cam_params: StereoCameraParams object
        Stereo camera parameters for stereo-setup used to capture 2D points
    num_planes: int
        Number of planes scanned by the laser for the pose recovery
    input_folder_path: str
        Path to the folder containing the 2D point npys saved from the laser scans.
        Input folder internal folder strucutre i assumed to be:
        input_folder_path/
            plane_{plane_num}_{axis}_uvs1.npy
            plane_{plane_num}_{axis}_uvs2.npy
    output_folder_path: str
        Path to the folder where the 3D point npys will be saved as
        output_folder_path/
            x_axis_scan_points3d.npy
            y_axis_scan_points3d.npy
            target_angle_points3d.npy
    
    Returns:
    x_axis_scan_points3d: np.array
        3D points corresponding to the laser scans on the x axis
    y_axis_scan_points3d: np.array
        3D points corresponding to the laser scans on the y axis
    target_angle_points3d: np.array
        3D points along the beam at a single angle during the scans, at different planes (depths)
    """
   
    # aggregate 3d points corresponding to laser scans on all planes
    x_axis_scan_points3d = None
    y_axis_scan_points3d = None
    for plane_num in range(num_planes): 
        if plane_num in skip_planes:
            continue
        for ax_num in range(2):
            if ax_num == 0: 
                axis = "x"
            elif ax_num == 1:  
                axis = "y"
        
            uvs1 = np.load(f"{input_folder_path}/plane_{plane_num}_{axis}_uvs1.npy")
            uvs2 = np.load(f"{input_folder_path}/plane_{plane_num}_{axis}_uvs2.npy")
            
            p3ds = triangulate(uvs1, uvs2, stereo_cam_params, display = False)

            if ax_num == 0:
                if x_axis_scan_points3d is None:
                    x_axis_scan_points3d = p3ds
                else:
                    x_axis_scan_points3d = np.concatenate((x_axis_scan_points3d, p3ds))
            elif ax_num == 1:
                if y_axis_scan_points3d is None:
                    y_axis_scan_points3d = p3ds
                else:
                    y_axis_scan_points3d = np.concatenate((y_axis_scan_points3d, p3ds))
    
    # also get the 3d points corresponding to the target angle
    target_angle_uvs1 = np.load(f"{input_folder_path}/target_angle_uvs1.npy")
    target_angle_uvs2 = np.load(f"{input_folder_path}/target_angle_uvs2.npy")
    target_angle_points3d = triangulate(target_angle_uvs1, target_angle_uvs2, stereo_cam_params, display = False)
    
    if output_folder_path is not None:
        np.save(f"{output_folder_path}/x_axis_scan_points3d.npy", x_axis_scan_points3d)
        np.save(f"{output_folder_path}/y_axis_scan_points3d.npy", y_axis_scan_points3d)
        np.save(f"{output_folder_path}/target_angle_points3d.npy", target_angle_points3d)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_axis_scan_points3d[:, 0], x_axis_scan_points3d[:, 1], x_axis_scan_points3d[:, 2], c='r', s=50, label = "X axis scan points")
    ax.scatter(y_axis_scan_points3d[:, 0], y_axis_scan_points3d[:, 1], y_axis_scan_points3d[:, 2], c='b', s=50, label = "Y axis scan points")
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim(-150, 200)
    # ax.set_ylim(-250, 50)
    # ax.set_zlim(300, 600)
    ax.set_title("Laser Scan Points\nCamera Reference Frame")
    plt.show()

    return x_axis_scan_points3d, y_axis_scan_points3d, target_angle_points3d


def recover_laser_basis(x_axis_scan_points3d, y_axis_scan_points3d):
    """
    Recover the laser's basis matrix from the 3D points captured durinng x and y axis scanes.
    Note that the basis does not capture information about the directions of the axes, only their relative orientations.
    For our purposes, the direction does not matter, since it only changes the sign of any ultimately computed laser angles.
    and we perform a mapping stage that make this a non-issue.

    Parameters:
    x_axis_scan_points3d : np.ndarray
        3D points captured during the x axis scan
    y_axis_scan_points3d : np.ndarray
        3D points captured during the y axis scan
    
    Returns:
    laser_B : 3x3 np.ndarray
        The laser's basis matrix
    intersection_line_points3d : list of 3x1 np.ndarray
        Two 3D points on the intersection line of the xz and yz tangent planes (i.e., the laser z axis), in the camera's reference frame. 
    """

    # fit planes to the x and y axis points captured at different planes during the scans
    func = func_degree2
    xz_surface_coeffs, X_xz_surface, Y_xz_surface, Z_xz_surface, rms = fit_3d_surface(x_axis_scan_points3d, func)
    yz_surface_coeffs, X_yz_surface, Y_yz_surface, Z_yz_surface, rms = fit_3d_surface(y_axis_scan_points3d, func)

    # get a point at the intersection of the two surfaces and the tangent planes at that point
    p = surface_surface_intersection_point(xz_surface_coeffs, yz_surface_coeffs, 400)
    p = np.array(p).reshape((3, 1))
    xz_tangent_eq = compute_tangent_plane(p, func, dx_func_degree2, dy_func_degree2, xz_surface_coeffs)
    yz_tangent_eq = compute_tangent_plane(p, func, dx_func_degree2, dy_func_degree2, yz_surface_coeffs)
 
    # get the intersection line of the two planes, which is the estimated laser z axis
    intersection_line_points3d = np.array(plane_plane_intersection(xz_tangent_eq, yz_tangent_eq))
    if intersection_line_points3d[0][2] > intersection_line_points3d[1][2]:
        z_axis_dir = Vector(intersection_line_points3d[0] - intersection_line_points3d[1])
    else:
        z_axis_dir = Vector(intersection_line_points3d[1] - intersection_line_points3d[0])

    # ensure z is positive
    if z_axis_dir[2] < 0:
        z_axis_dir = -1 * z_axis_dir
    z_axis = Line(point = Point(intersection_line_points3d[0]), direction = z_axis_dir)

    # estimate the laser x axis by finding the line that is orthogonal to the z axis and lies on the estimated XZ plane
    A = np.array([[xz_tangent_eq[0], xz_tangent_eq[1]], 
                [z_axis.direction[0], z_axis.direction[1]]])
    b = np.array([-xz_tangent_eq[2], -z_axis.direction[2]])
    sol = np.linalg.solve(A, b)
    x_axis_dir = [sol[0], sol[1], 1] # set z component to 1 arbitrarily, so that x_axis_dir is the ratio of the x and y components
    # ensure x is positive
    if x_axis_dir[0] < 0:
        x_axis_dir = -1 * np.array(x_axis_dir)
    x_axis = Line(point = Point(z_axis.point), direction = Vector(x_axis_dir)) # choose arbitrary point in z axis as start of the x axis (i.e., origin), as this is currently unknown
    
    # estimate the laser y axis, which is just the normal XZ plane, as the x axis has been assumed to lie in this plane
    y_axis_dir = xz_tangent_eq[:3]
    # ensure y is positive
    if y_axis_dir[1] < 0:
        y_axis_dir = -1 * np.array(y_axis_dir)
    y_axis = Line(point = Point(z_axis.point), direction = y_axis_dir)

    # sanity checks
    print("\nSanity checks - Below numbers should all be ~90:")
    x_z_angle = get_angle(x_axis.direction, z_axis.direction)
    print("Angle between x and z axes:", x_z_angle)
    y_z_angle = get_angle(y_axis.direction, z_axis.direction)
    print("Angle between y and z axes:", y_z_angle)
    X_yz_surface_angle = get_angle(x_axis.direction, y_axis.direction)
    print("Angle between x and y axes:", X_yz_surface_angle)
    tangent_plane_angles = get_angle(xz_tangent_eq[:3], yz_tangent_eq[:3])
    print("Angle between tangent planes (should be close to 90 deg hopefully):", tangent_plane_angles)

  
    # norm the axis directions and combine them into the laser's basis matrix
    x_axis_vec  = x_axis.direction /  np.linalg.norm(x_axis.direction)
    y_axis_vec = y_axis.direction / np.linalg.norm(y_axis.direction)
    z_axis_vec = z_axis.direction / np.linalg.norm(z_axis.direction)
    x_axis_vec = np.array(x_axis_vec)
    y_axis_vec = np.array(y_axis_vec)
    z_axis_vec = np.array(z_axis_vec)
    laser_B = np.hstack((x_axis_vec.reshape((3, 1)), y_axis_vec.reshape((3, 1)), z_axis_vec.reshape((3, 1))))


    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')

    x_axis_scan_points_transformed = transform_cam2laser(x_axis_scan_points3d, LaserPoseParams(laser_B))
    y_axis_scan_points_transformed = transform_cam2laser(y_axis_scan_points3d, LaserPoseParams(laser_B))

    ax.scatter(x_axis_scan_points_transformed[:, 0], x_axis_scan_points_transformed[:, 1], x_axis_scan_points_transformed[:, 2], c="#57009a", alpha = 1, label = "X axis scan points")
    ax.scatter(y_axis_scan_points_transformed[:, 0], y_axis_scan_points_transformed[:, 1], y_axis_scan_points_transformed[:, 2],  c="#57009a", alpha = 1, label = "Y axis scan points")

    xx, yy = np.meshgrid(range(-100, 150), range(-5, 5))
    xz_tangent_z = (-1*xz_tangent_eq[3] - xz_tangent_eq[0] * xx - xz_tangent_eq[1] * yy) * 1. / xz_tangent_eq[2]
    ps = np.hstack((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1), xz_tangent_z.ravel().reshape(-1, 1)))
    ps_prime = transform_cam2laser(ps, LaserPoseParams(laser_B))  
    xx_prime = ps_prime[:, 0].reshape(xx.shape) 
    yy_prime = ps_prime[:, 1].reshape(yy.shape)
    xz_tangent_z_prime = ps_prime[:, 2].reshape(xz_tangent_z.shape)
    ax.plot_surface(xx_prime, yy_prime, xz_tangent_z_prime, color='grey', alpha = 0.4, label = "Tangent planes")

    xx, yy = np.meshgrid(range(-50, 100), range(-200, 200))
    yz_tangent_z = (-1*yz_tangent_eq[3] - yz_tangent_eq[0] * xx - yz_tangent_eq[1] * yy) * 1. / yz_tangent_eq[2]
    ps = np.hstack((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1), yz_tangent_z.ravel().reshape(-1, 1)))
    ps_prime = transform_cam2laser(ps, LaserPoseParams(laser_B))
    xx_prime = ps_prime[:, 0].reshape(xx.shape)
    yy_prime = ps_prime[:, 1].reshape(yy.shape)
    yz_tangent_z_prime = ps_prime[:, 2].reshape(yz_tangent_z.shape)
    ax.plot_surface(xx_prime, yy_prime, yz_tangent_z_prime, color='grey', alpha = 0.4)

    # transformed_z = Line(point = Point(z_axis.point), direction = Vector((0, 0, 1)))
    # transformed_z.plot_3d(ax, t_1 = -200, t_2 = 1000, c='black', alpha = 0.5, label = "Estimated laser Z axis")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.grid(False)
    # plt.axis('off')
    ax.view_init(elev=0, azim=323, roll=0)
   
    plt.show()
    # save figure
    # fig.savefig("fig.png", dpi=800)
    # plt.clf()

    ################################################ PLOTTING ################################################
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # first, the scan points, the fitted surfaces, and the surfaces' intersection poin
    ax.scatter(x_axis_scan_points3d[:, 0], x_axis_scan_points3d[:, 1], x_axis_scan_points3d[:, 2], c='r', s=50, label = "X axis scan points")
    ax.scatter(y_axis_scan_points3d[:, 0], y_axis_scan_points3d[:, 1], y_axis_scan_points3d[:, 2], c='b', s=50, label = "Y axis scan points")
    ax.plot_surface(X_xz_surface, Y_xz_surface, Z_xz_surface, color='r', alpha = 0.5, label = "XZ fitted surface")
    ax.plot_surface(X_yz_surface, Y_yz_surface, Z_yz_surface, color='b', alpha = 0.5, label = "YZ fitted surface")
    # ax.scatter(p[0], p[1], p[2], c='g', s=50) # intersection point

    # then plot the tangent planes and their intersection line points
    xz_tangent_z = (-1*xz_tangent_eq[3] - xz_tangent_eq[0] * X_xz_surface - xz_tangent_eq[1] * Y_xz_surface) * 1. / xz_tangent_eq[2]
    ax.plot_surface(X_xz_surface, Y_xz_surface, xz_tangent_z, color='grey', alpha = 0.4, label = "Tangent planes")
    yz_tangent_z = (-1*yz_tangent_eq[3] - yz_tangent_eq[0] * X_yz_surface - yz_tangent_eq[1] * Y_yz_surface) * 1. / yz_tangent_eq[2]
    ax.plot_surface(X_yz_surface, Y_yz_surface, yz_tangent_z, color='grey', alpha = 0.4)
    ax.scatter(intersection_line_points3d[:, 0], intersection_line_points3d[:, 1], intersection_line_points3d[:, 2], c='black', s=50) # intersection line points

    #plot x, y, and z axes
    max_z = max(np.max(x_axis_scan_points3d[:, 2]), np.max(y_axis_scan_points3d[:, 2])) # use this to determine a good length of the axes in plot
    z_target_t = (max_z - z_axis.point[2]) / z_axis.direction[2]
    x_target_t = (max_z - x_axis.point[2]) / x_axis.direction[2]
    y_target_t = (max_z - y_axis.point[2]) / y_axis.direction[2]
    z_axis.plot_3d(ax, t_1 = 0, t_2 = 1, c='green', alpha = 0.5, label = "Estimated laser Z axis")
    x_axis.plot_3d(ax, t_1= 0, t_2 = 1, c='red', label = "Estimated laser X axis")
    y_axis.plot_3d(ax, t_1= 0, t_2 = 1, c='blue', label = "Estimated laser Y axis")

    ax.axis('equal')
    # ax.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    return laser_B, intersection_line_points3d


def visualize_transformation(x_axis_scan_points3d, y_axis_scan_points3d, laser_pose_params):
    """
    Visualize the transformation of the x and y axis scan points from camera to laser reference frame
    given the laser's pose parameters, i.e., basis (rotation) and translation relative to the camera origin in the camera's basis.

    Parameters:
    x_axis_scan_points3d : list of 3x1 np.ndarray
        3D points captured during the x axis scan, in the camera's reference frame
    y_axis_scan_points3d : list of 3x1 np.ndarray
        3D points captured during the y axis scan, in the camera's reference frame
    laser_pose_params : LaserPoseParams object
        The laser's pose parameters, i.e., basis (rotation) and translation relative to the camera origin in the camera's basis
    
    Returns:
    None
    """
 
    fig = plt.figure()
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(right=0.8)

    # original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x_axis_scan_points3d[:, 0], x_axis_scan_points3d[:, 1], x_axis_scan_points3d[:, 2], c='r', s=50, label = "X axis scan points")
    ax1.scatter(y_axis_scan_points3d[:, 0], y_axis_scan_points3d[:, 1], y_axis_scan_points3d[:, 2], c='b', s=50, label = "Y axis scan points")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title("Points in Cam\nReference Frame")
    ax1.legend()
    ax1.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=7)

    # transformed 
    x_axis_scan_points3d_transformed = transform_cam2laser(x_axis_scan_points3d, laser_pose_params)
    y_axis_scan_points3d_transformed = transform_cam2laser(y_axis_scan_points3d, laser_pose_params)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x_axis_scan_points3d_transformed[:, 0], x_axis_scan_points3d_transformed[:, 1], x_axis_scan_points3d_transformed[:, 2], c='r', s=50, label = "X axis scan points")
    ax2.scatter(y_axis_scan_points3d_transformed[:, 0], y_axis_scan_points3d_transformed[:, 1], y_axis_scan_points3d_transformed[:, 2], c='b', s=50, label = "Y axis scan points")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title("Transformed Points")
    ax2.legend()
    ax2.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=7)

    fig.suptitle("Laser Transformation Visualization")
    fig.tight_layout()
    plt.show()

def recover_laser_translation(laser_B, intersection_line_points3d, target_angle_points3d, x_axis_scan_points3d, y_axis_scan_points3d, target_angle_axis = "x"):
    """
    Return translations of laser origin relative to the camera origin, in the camera's basis, i.e., standard basis with camera origin at (0, 0, 0).

    Parameters:
    laser_B : 3X3 np.ndarray
        The laser's basis (pose R) matrix, with the x, y, and z axes as columns
    intersection_line_points3d : list of 3x1 np.ndarray
        Two 3D points on the intersection line of the xz and yz tangent planes (i.e., the laser z axis), in the camera's reference frame. 
    target_angle_points3d : list of 3x1 np.ndarray
        3D points along the laser beam at the target angle, in the camera's reference frame
    x_axis_scan_points3d : list of 3x1 np.ndarray
        3D points captured during the x axis scan, in the camera's reference frame
    y_axis_scan_points3d : list of 3x1 np.ndarray
        3D points captured during the y axis scan, in the camera's reference frame
    target_angle_axis : str
        The axis along which the laser was steering at the target angle. Either "x" or "y". Default is "x",
        as currently, get_laser_pose_recovery_points2d() uses images for an x axis scan angle as the target_angle.
    
    Returns:
    laser_translation : list
        Translation of the laser origin relative to the camera origin in the camera

    Algorithm:
    1. Get the laser basis matrix from the laser's x, y, and z axes
    2. Get the x and y translation of the laser origin relative to the camera origin in the laser's basis
    3. Express all the points along the beam at the target angle in the laser's basis, shifted by the translations, and fit a line to them
    4. Get the points along this line with approximately the largest and smallest z values, and use these to define a triangle 
        where the segment connecting them is the triangle's hypotenuse; the segment roughly parallel to the x or y axis -- whichever 
        axis the laser was steering along at the target angle, as specified by the target_angle_dir parameter -- connecting the min and max 
        point on line's x or y coords is the triangle's base; and the z delta between the min and max point on line is the triangle's height
    5. Define a similar triangle whose base connects the min point on line to the z axis, and whose other side is colinear with the z axis. 
        Since all points on the beam emanate from the laser origin, the height of this similar triangle gives the absolute z coord of the min point 
        on line in the laser's reference frame. Let's call this z_target.
    6. Solve for z_target using the similar triangles property, and the base length and height of the reference triangle and the base of the target triangle
        as the "knowns."
    7. Subtract z_target from the z coord of the min point on line to get the z translation of the laser origin relative to the camera origin
        in the laser's basis
    8. Combinging results from 1. and 5., we have the x, y, and z translations of the laser origin relative to the camera origin in the laser's basis.
        Since the camera origin is just (0, 0, 0), the translations are simply a coordinate (x_translation, y_translation, z_translation) in the 
        camera reference frame upon being rotated by the laser's basis matrix.
    9. Apply the basis change (just laser basis without transpose) to (x_translation, y_translation, z_translation) to get the laser's position (and thus translation)
        relative to the camera origin in the camera's basis.
    """
    
    print("Laser basis matrix:")
    print(laser_B)

    # apply basis change to intersection of x and y axes points.
    # either of their x and y coords (should be equivalent for each transformed point, with only z coord differing) 
    # gives x, y translation of the laser origin relative to camera origin in the laser's basis
    intersection_line_points3d = np.array(intersection_line_points3d)
    intersection_line_points3d_transformed = np.matmul(laser_B.T, intersection_line_points3d.T).T


    laser_x_translation_B = intersection_line_points3d_transformed[0][0]
    laser_y_translation_B = intersection_line_points3d_transformed[0][1] 
    print("Laser origin x translation relative to camera origin, laser basis:", laser_x_translation_B)
    print("Laser origin y translation relative to camera origin, laser basis:", laser_y_translation_B)
    
    # transform the points along the laser beam at target angle -- including basis change and x and y translations -- and fit a line to them
    # again important to subtract the translations from the x and y coords of the points prior to rotation, as the translations are relative to the camera axes
    target_angle_points3d_transformed = target_angle_points3d.copy()
    target_angle_points3d_transformed = np.matmul(laser_B.T, target_angle_points3d_transformed.T).T 
    target_angle_points3d_transformed[:, 0] = target_angle_points3d_transformed[:, 0] - laser_x_translation_B
    target_angle_points3d_transformed[:, 1] = target_angle_points3d_transformed[:, 1] - laser_y_translation_B
    line = Line.best_fit(target_angle_points3d_transformed)

    # get the points along this line with the largest and smallest z values,
    # and query the line equation at these points to get the points on the line at these z values, for use in creating the similar triangles below
    max_z_idx = np.argmax(np.abs(target_angle_points3d_transformed[:, 2])) # important to abs the z values, just in case the z axis is flipped
    min_z_idx = np.argmin(np.abs(target_angle_points3d_transformed[:, 2]))
    min_z_point = target_angle_points3d_transformed[min_z_idx]
    max_z_point = target_angle_points3d_transformed[max_z_idx]
    
    min_z_point_on_line_t = (line.point[2] - min_z_point[2]) / line.direction[2]
    max_z_point_on_line_t = (line.point[2] - max_z_point[2]) / line.direction[2]
    min_z_point_on_line = line.point + min_z_point_on_line_t * line.direction
    max_z_point_on_line = line.point + max_z_point_on_line_t * line.direction
    # possible that they may flip order if the line is not oriented as expected depending on line.direction signage,
    # but that's fine as the points are just used to define a segment and we use absolute valued differences below
    # correct order for later visualization purposes, though
    if np.abs(min_z_point_on_line[2]) > np.abs(max_z_point_on_line[2]): # again important to abs in case of flipped z axis
        temp = min_z_point_on_line
        min_z_point_on_line = max_z_point_on_line
        max_z_point_on_line = temp

    # get the the reference similar triangle's side dimensions, using x or y coords for definining target base, and solve for the target triangle's height
    if target_angle_axis == "x":
        base_ref = np.abs(max_z_point_on_line[0] - min_z_point_on_line[0])
    else:
        base_ref = np.abs(max_z_point_on_line[1] - min_z_point_on_line[1])
    if target_angle_axis == "x":
        base_target = np.abs(min_z_point_on_line[0]) #abs because depending on direction of laser steer at target angle, the min z point on line may be on either left or right side the z axis
    else:
        base_target = np.abs(min_z_point_on_line[1]) # since min_z_point_on_line point has been rotated and translated, its x/y value is the x/y value of the larger triangle, with its base ~parallel the laser's x/y axis
    z_ref = np.abs(np.abs(max_z_point_on_line[2]) - np.abs(min_z_point_on_line[2]))
    z_target = (z_ref * base_target) / base_ref # solve for the triangle's height, z_target, using similar triangles property. This gives the actual z coord of min_z_point_on_line in the laser's reference frame and relative to the steering device's origin

    laser_origin_B = np.array([0, 0, min_z_point_on_line[2] - z_target]) # the laser origin in the laser's basis, before correcting for the z translation. 
    # subtract z target above because it extends to the origin of the laser steering device, which would have a smaller z axis value than any points along beam
    print("Laser origin z translation relative to camera origin, laser basis:", min_z_point_on_line[2] - z_target)
    laser_origin_B_untranslated = [laser_x_translation_B, laser_y_translation_B, laser_origin_B[2]] # un-apply the x and y translations to get the laser origin in the laser's basis, relative to the camera origin
    laser_origin_cam = np.matmul(laser_B, laser_origin_B_untranslated) # get the laser origin in the camera's reference frame
    print("Laser origin in camera basis:", laser_origin_cam)
    laser_x_translation, laser_y_translation, laser_z_translation = laser_origin_cam # since the laser origin is the origin of the laser's reference frame, the laser's translation relative to the camera origin is just the laser origin's coords in the camera's reference frame
    laser_translation = [laser_x_translation, laser_y_translation, laser_z_translation]

    ################################################ PLOTTING ################################################

    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(right=0.8)
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x_axis_scan_points3d[:, 0], x_axis_scan_points3d[:, 1], x_axis_scan_points3d[:, 2], c='r', s=50, alpha=0.2, label = "All x axis scan points")
    ax1.scatter(y_axis_scan_points3d[:, 0], y_axis_scan_points3d[:, 1], y_axis_scan_points3d[:, 2], c='b', s=50, alpha=0.2, label = "All y axis scan points")
    ax1.scatter(target_angle_points3d[:, 0], target_angle_points3d[:, 1], target_angle_points3d[:, 2], c='g', s=50, alpha=0.5, label = "Target beam points")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=7)
    ax1.set_title("Points in Cam\nReference Frame")

    # transformed points and triangles visualization
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(target_angle_points3d_transformed[:, 0], target_angle_points3d_transformed[:, 1], target_angle_points3d_transformed[:, 2], c='g', s=50, alpha=0.2, label = "Target beam points")
    ax2.scatter(min_z_point_on_line[0], min_z_point_on_line[1], min_z_point_on_line[2], c='g', s=50, label = "Target beam points on line")
    ax2.scatter(max_z_point_on_line[0], max_z_point_on_line[1], max_z_point_on_line[2], c='g', s=50)
    line.plot_3d(ax2, t_1=-200, t_2=200, c='g', linestyle = '--', label="Beam best fit line")
    # x_axis_scan_points3d_transformed = np.matmul(laser_B.T, x_axis_scan_points3d.T).T
    # ax2.scatter(x_axis_scan_points3d_transformed[:, 0], x_axis_scan_points3d_transformed[:, 1], x_axis_scan_points3d_transformed[:, 2], c='r', s=50, alpha=0.2, label = "All x axis scan points")
    
    if target_angle_axis == "x": 
        ax2.plot([min_z_point_on_line[0], min_z_point_on_line[0]], [min_z_point_on_line[1], min_z_point_on_line[1]], [max_z_point_on_line[2], min_z_point_on_line[2]], c='g', alpha = 0.5, label="Ref. Triangle Sides") # height of small triangle
        ax2.plot([max_z_point_on_line[0], min_z_point_on_line[0]], [min_z_point_on_line[1], min_z_point_on_line[1]], [max_z_point_on_line[2], max_z_point_on_line[2]], c='g', alpha = 0.5) # dont bother reentering legend description
        ax2.plot([0, min_z_point_on_line[0]], [0, min_z_point_on_line[1]], [min_z_point_on_line[2], min_z_point_on_line[2]], c='Magenta', alpha = 0.5, label="Target Triangle Sides")
    elif target_angle_axis == "y":
        ax2.plot([min_z_point_on_line[0], max_z_point_on_line[0]], [min_z_point_on_line[1], min_z_point_on_line[1]], [max_z_point_on_line[2], min_z_point_on_line[2]], c='g', alpha = 0.5, label="Ref. Triangle Sides") # height of small triangle
        ax2.plot([max_z_point_on_line[0], min_z_point_on_line[0]], [max_z_point_on_line[1], min_z_point_on_line[1]], [max_z_point_on_line[2], max_z_point_on_line[2]], c='g', alpha = 0.5)
        ax2.plot([0, max_z_point_on_line[0]], [0, min_z_point_on_line[1]], [min_z_point_on_line[2], min_z_point_on_line[2]], c='Magenta', alpha = 0.5, label="Target Triangle Sides")
  
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_ylim(-50, 50)
    ax2.legend()
    ax2.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=7)
    ax2.set_title("Points in Laser Reference Frame\n(w/ Arbitrary Z Translation)")

    fig.suptitle("Laser Pose Translation Recovery Visualization")
    fig.tight_layout()
    plt.show()

    # visualize our final result
    fig = plt.figure()
    fig.subplots_adjust(right=0.8)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_axis_scan_points3d[:, 0], x_axis_scan_points3d[:, 1], x_axis_scan_points3d[:, 2], c='r', s=50, alpha=0.2, label = "All x axis scan points")
    ax.scatter(y_axis_scan_points3d[:, 0], y_axis_scan_points3d[:, 1], y_axis_scan_points3d[:, 2],  c='b', s=50, alpha=0.2, label = "All y axis scan points")
    ax.scatter(laser_origin_cam[0], laser_origin_cam[1], laser_origin_cam[2], c='magenta', s=50, label = "Recovered laser origin")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=7)
    ax.set_title("Final Result\nPoints in Cam Reference Frame")
    fig.tight_layout()
    plt.show()

    return laser_translation

def visualize_poses(stereo_cam_params, laser_params):
    """
    Visualizes the poses of multiple devices in 3D space. 
    Informed by: https://stackoverflow.com/questions/8178467/how-to-plot-the-camera-and-image-positions-from-camera-calibration-data
    """

    identity_pose = np.eye(4)
    identity_pose[0, 0] = 1
    identity_pose[1, 1] = 1
    identity_pose[2, 2] = 1

    print(stereo_cam_params.R)
    print(stereo_cam_params.T)
    T_inv = stereo_cam_params.T.reshape((3, 1))
    T_inv[:2] = -1 * T_inv[:2] # don't invert z axis by convention
    cam2_pose = np.hstack((stereo_cam_params.R.T, T_inv))

    laser_pose = np.hstack((laser_params.pose_R, laser_params.pose_T.reshape((3, 1))))
    print(laser_pose)

    poses = [identity_pose, cam2_pose, laser_pose]

    cam_origin_colors = ['orange', 'cyan', 'magenta']
    ax = plt.figure().add_subplot(projection='3d')
    ax_labeled = False

    for cam_num, camera_extrinsics in enumerate(poses):
        # Extract translation and rotation from camera extrinsics matrix
        translation = camera_extrinsics[:3, 3]
        rotation_matrix = camera_extrinsics[:3, :3]

        # Plot camera position
        ax.scatter(*translation, marker='o', c = cam_origin_colors[cam_num])

        # Plot camera orientation axes
        origin = translation
        for i in range(3):
            axis_direction = rotation_matrix[:,i] 
            if i == 0:
                if not ax_labeled:
                    ax.quiver(*origin, *axis_direction, length=10, normalize=True, color = "g", label = "X axis")
                else:
                    ax.quiver(*origin, *axis_direction, length=10, normalize=True, color = "g")
            elif i == 1:
                if not ax_labeled and i == 1:
                    ax.quiver(*origin, *axis_direction, length=10, normalize=True, color = "b", label = "Y axis")
                else:
                    ax.quiver(*origin, *axis_direction, length=10, normalize=True, color = "b")
            else:
                if not ax_labeled:
                    ax.quiver(*origin, *axis_direction, length=10, normalize=True, color='r', alpha=0.5, label = "Z axis")
                else:
                    ax.quiver(*origin, *axis_direction, length=10, normalize=True, color='r', alpha=0.5)
        ax_labeled = True
        text_loc = [translation[0], translation[1], translation[2] + 10]
        if cam_num == 2:
            dev_name = "Laser"
        else:
            dev_name = f"Cam {cam_num + 1}"
        ax.text(*text_loc, dev_name, c = cam_origin_colors[cam_num])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Cameras + Laser Poses")
    ax.axis('equal')
    ax.legend()
    plt.show()


#########
# MAPPING
#########
def get_mapping_points2d(images_path, stereo_cam_params, manual_approve = False, manual_selection = False, output_folder_path = "."):
    """
    Get the laser spots from images taken durring spiral scan for mapping
    
    Parameters:
    images_path : str
        Path to folder containing stereocam images of scans on multiple planes, where the internal folder structure is assumed to be: 
        images_path/
            LEFT/
                *.png
            RIGHT/
                *.png   
    stereo_params_path : str
        Path to pickle file containing stereo camera calibration parameters
    manual_approve : bool
        If True, will require user input to approve the detected laser spots in each image. 
        If False, will automatically select the brightest spot in each image.
    manual_selection : bool
        If True, will require user input to click on the laser spot in each image. 
    output_folder_path : str
        Path to folder to save .npy files containing the detected laser spots.
    """

    if not os.path.exists(images_path):
        raise ValueError("Invalid images path")


    if manual_approve and manual_selection:
        raise ValueError("Cannot have both manual_approve and manual_selection set to True")
        

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    imgs_left = glob.glob(f"{images_path}/LEFT/*.png")
    imgs_right = glob.glob(f"{images_path}/RIGHT/*.png")
    imgs_left.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]))
    imgs_right.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]))

    uvs1 = []
    uvs2 = []
    steers = [] # steering input associated with each uv pair
    for i in range(len(imgs_left)):
        
        if manual_selection:
            left_blob_selections, right_blob_selections, left_frame, right_frame = select_test_correspondences(imgs_left[i], imgs_right[i], stereo_cam_params, undistort_frames=True)
            if len(left_blob_selections) == 0 or len(right_blob_selections) == 0:
                print("No blobs selected")
                continue
            
            uvs1.append(left_blob_selections[0])
            uvs2.append(right_blob_selections[0])
        else:

            left_img = cv2.imread(imgs_left[i])
            right_img = cv2.imread(imgs_right[i])

            # undistort the images
            left_img = undistort(left_img, stereo_cam_params.cam1_params)
            right_img = undistort(right_img, stereo_cam_params.cam2_params)
   
            # detect blob in each
            left_blob_obj = get_brightest_contour(left_img, otsu_adjustment=10, contour_area_threshold=1000000)
            left_img_vis = left_img.copy()
            if left_blob_obj is not None:
                # draw blob on left img
                cv2.circle(left_img_vis, left_blob_obj["center"], left_blob_obj["radius"], (0, 255, 0), 2)
            else:
                print("No blob detected in left frame")    
            
            right_blob_obj = get_brightest_contour(right_img, otsu_adjustment=10, contour_area_threshold=1000000)
            right_img_vis = right_img.copy()
            if right_blob_obj is not None:
                # draw blob on right img
                cv2.circle(right_img_vis, right_blob_obj["center"], right_blob_obj["radius"], (0, 255, 0), 2)
            else:
                print("No blob detected in right frame")
            
        
            disp = np.hstack((left_img_vis, right_img_vis))
            cv2.imshow("", disp)
           
            if manual_approve:
                # wait for user decision to add or not
                k = cv2.waitKey(0)
                if k == ord('q'):
                    break
                elif k != ord('a'):
                    continue
                else:
                    if left_blob_obj is None or right_blob_obj is None:
                        continue
            else:
                if cv2.waitKey(1) == ord('q'):
                    break
                # automatically add if blob detected in both images
                if left_blob_obj is None or right_blob_obj is None:
                    continue
                    
            uvs1.append(left_blob_obj["center"])
            uvs2.append(right_blob_obj["center"])
            x_steer = imgs_left[i].split("/")[-1].split("_x")[1].split("_")[0]
            y_steer = imgs_left[i].split("/")[-1].split("_y")[1].split(".png")[0]
            print(i, x_steer, y_steer)
            steers.append((float(x_steer), float(y_steer)))

    np.save(f"{output_folder_path}/mapping_uvs1.npy", uvs1)
    np.save(f"{output_folder_path}/mapping_uvs2.npy", uvs2)
    np.save(f"{output_folder_path}/mapping_steers.npy", steers)
    print("Saved laser spots")
     
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)



def get_mapping_functions(stereo_cam_params, laser_params, input_folder_path = ".", output_folder_path = "."):
    """
    Get the functions to map outgoing laser angles (theta and phi) to laser steering inputs, given spiral scan points from the stereo camera. 

    Parameters:
    stereo_cam_params: StereoCameraParams object
        Stereo camera parameters
    laser_params: LaserParams object
        Laser pose parameters
    input_folder_path: str
        Path to folder containing spiral scan 2D points and associated steering inputs
        Internal folder structure is assumed to be:
            input_folder_path
            ├── mapping_uvs1.npy
            ├── mapping_uvs2.npy
            ├── mapping_steers.npy
    output_folder_path: str
        Path to folder to save mapping functions
        Internal folder structure will be:
            output_folder_path
            ├── mir_x_popt.npy
            ├── mir_y_popt.npy
        
    Returns:
    None
    """

    uvs1 = np.load(f"{input_folder_path}/mapping_uvs1.npy")
    uvs2 = np.load(f"{input_folder_path}/mapping_uvs2.npy")   
    steers = np.load(f"{input_folder_path}/mapping_steers.npy")    
    p3ds = triangulate(uvs1, uvs2, stereo_cam_params)    
    p3ds_transformed = transform_cam2laser(p3ds, laser_params)
    np.save(f"{output_folder_path}/mapping_p3ds.npy", p3ds_transformed)

    # for each transformed point, get its theta and phi, which are angles in the x and y planes, respectively
    thetas = []
    phis = []
    for i in range(p3ds_transformed.shape[0]):
        p3d = p3ds_transformed[i]
        theta = np.rad2deg(np.arctan2(p3d[0], p3d[2]))
        phi = np.rad2deg(np.arctan2(p3d[1], p3d[2]))
        thetas.append(theta)
        phis.append(phi)
    
    
    # map thetas and phis to the steering inputs
    mir_x_data = np.hstack((np.array(thetas).reshape(-1, 1), np.array(phis).reshape(-1, 1), np.array(steers)[:, 0].reshape(-1, 1)))
    mir_x_popt, X_mir_x_surface, Y_mir_x_surface, Z_mir_x_surface, rms_x = fit_3d_surface(mir_x_data, func_degree2)
    mir_y_data = np.hstack((np.array(thetas).reshape(-1, 1), np.array(phis).reshape(-1, 1), np.array(steers)[:, 1].reshape(-1, 1)))
    mir_y_popt, X_mir_y_surface, Y_mir_y_surface, Z_mir_y_surface, rms_y = fit_3d_surface(mir_y_data, func_degree2)

    print("Mirror x RMS:", rms_x)
    print("Mirror y RMS:", rms_y)
    np.save(f"{output_folder_path}/mir_x_popt.npy", mir_x_popt)
    np.save(f"{output_folder_path}/mir_y_popt.npy", mir_y_popt)
    
    ################################################ PLOTTING ################################################
    # spiral scan points
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(p3ds[:, 0], p3ds[:, 1], p3ds[:, 2], c='r', marker='o')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.axis('equal')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(p3ds_transformed[:, 0], p3ds_transformed[:, 1], p3ds_transformed[:, 2], c='b', marker='o')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.axis('equal')

    fig.tight_layout()
    fig.suptitle("Mapping Spiral Scan Points\nLaser Reference Frame")
    plt.show()

    # sanity check visualization that thetas/phis are reasonable
    # fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # vis = 80
    # for i in range(p3ds_transformed.shape[0]):
    #     if i != vis:
    #         continue
    #     ax.scatter(p3ds_transformed[i, 0], p3ds_transformed[i, 1], p3ds_transformed[i, 2], c='r', marker='o')
    #     ax.plot([0, p3ds_transformed[i, 0]], [0, p3ds_transformed[i, 1]], [0, p3ds_transformed[i, 2]], c='r')
    #     ax.text(p3ds_transformed[i, 0], p3ds_transformed[i, 1], p3ds_transformed[i, 2], f"theta: {thetas[i]:.2f}, phi: {phis[i]:.2f}")
    #     ax.text(p3ds_transformed[i, 0] - 5, p3ds_transformed[i, 1] - 5, p3ds_transformed[i, 2] - 200, f"coords: ({p3ds_transformed[i, 0]:.0f}, {p3ds_transformed[i, 1]:.0f}, {p3ds_transformed[i, 2]:.0f})")
    # ax.set_xlabel('X')  
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.axis('equal')
    # plt.show()

    # mapping fit 
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(mir_x_data[:, 0], mir_x_data[:, 1], mir_x_data[:, 2] , c='r', marker='o')
    ax1.plot_surface(X_mir_x_surface, Y_mir_x_surface, Z_mir_x_surface, color='r', alpha=0.5)
    ax1.set_xlabel('Theta')
    ax1.set_ylabel('Phi')
    ax1.set_zlabel('Mir x')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(mir_y_data[:, 0], mir_y_data[:, 1], mir_y_data[:, 2] , c='b', marker='o')
    ax2.plot_surface(X_mir_y_surface, Y_mir_y_surface, Z_mir_y_surface, color='b', alpha=0.5)
    ax2.set_xlabel('Theta')
    ax2.set_ylabel('Phi')
    ax2.set_zlabel('Mir y')

    fig.tight_layout()
    fig.suptitle("Laser Mappings")
    plt.show()