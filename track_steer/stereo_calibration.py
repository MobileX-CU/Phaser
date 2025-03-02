"""
All functions and classes needed to calibrate the stereo-camera setup, 
including intrinsics and extrinsics for each camera, to facilitate
triangulation, disparity mapping, and undistortion
"""
import os
import cv2
import numpy as np
import glob
from enum import Enum
import pickle

class CamType(Enum):
    NORMAL = 1
    FISHEYE = 2

class SingleCameraParameters():
    def __init__(self, K, D, cal_dims, cam_type):
        self.K = K
        self.D = D
        self.cal_dims = cal_dims
        self.camType = cam_type

class Calibrator():
    def __init__(self, checkerboard_dims = (6, 9), square_size = 1, criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
                 cam_type = CamType.NORMAL):
        self.checkerboard_dims = checkerboard_dims
        self.square_size = square_size
        self.criteria = criteria
        self.cam_type = cam_type
    
    def get_calibration_points(self, frames_path, annotated_frames_path = None):
        """
        Largely from https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
        """

        if annotated_frames_path is not None:
            os.makedirs(annotated_frames_path, exist_ok = True)
           
        # 3d coordiantes for checkerboard vertices in real world space
        objp = np.zeros((self.checkerboard_dims[0]*self.checkerboard_dims[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.checkerboard_dims[0],0:self.checkerboard_dims[1]].T.reshape(-1,2)
        objp = self.square_size * objp

        imgpoints = [] # 2d vertices in image plane.
        objpoints = [] # 3d checkerboard vertices in real world space, to associate with each detected 2d vertex
        
        frame_paths = glob.glob(frames_path + "/*.png")
        frame_paths.sort(key=lambda f: int(f.split("/")[-1].split(".")[0]))

        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #find the checkerboard
            corners_detected, corners = cv2.findChessboardCorners(gray, self.checkerboard_dims, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if corners_detected:
                #Convolution size used to improve corner detection. Don't make this too large.
                conv_size = (11, 11)
                corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), self.criteria)
                annotated_frame = frame.copy()
                cv2.drawChessboardCorners(annotated_frame,self.checkerboard_dims, corners, corners_detected)
                cv2.imshow('Corner Detection', annotated_frame)
                if cv2.waitKey(1) == ord('q'): break
                objpoints.append(objp)
                imgpoints.append(corners)
                if annotated_frames_path is not None:
                    cv2.imwrite(f"{annotated_frames_path}/{i}.png", annotated_frame)
            else:
                objpoints.append(None) # important to have record of None for stereo calibration
                imgpoints.append(None)
                cv2.imshow('Corner Detection', frame)
                if cv2.waitKey(1) == ord('q'): break
 
        return objpoints, imgpoints, gray.shape[::-1]


class SingleCameraCalibrator(Calibrator):
    def __init__(self, frames_path, checkerboard_dims = (6, 9), square_size = 1, criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
                 cam_type = CamType.NORMAL):
        super().__init__(checkerboard_dims, square_size, criteria, cam_type)
        self.frames_path = frames_path
      
    def calibrate(self, annotated_frames_path = None):
        print("Calibrating individual camera from objpoints and imgpoints...")
        objpoints, imgpoints, imgs_dim = self.get_calibration_points(self.frames_path, annotated_frames_path=annotated_frames_path)
        # remove entries where the entry is np.nan
        objpoints = np.array([objpoints[i] for i in range(len(objpoints)) if objpoints[i] is not None])
        imgpoints = np.array([imgpoints[i] for i in range(len(imgpoints)) if imgpoints[i] is not None])
        if self.cam_type == CamType.NORMAL:
            return self.generate_intrinsics(objpoints, imgpoints, imgs_dim)
        elif self.cam_type == CamType.FISHEYE:
            return self.generate_fisheye_intrinsics(objpoints, imgpoints, imgs_dim)
        else:
            raise ValueError("Invalid camera type.")
        
    def generate_intrinsics(self, objpoints, imgpoints, imgs_dim):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (imgs_dim[0], imgs_dim[1]), None, None, flags = cv2.CALIB_USE_LU)
        print("Intrinsic Calibration RMS:", ret) 
        return SingleCameraParameters(mtx, dist, imgs_dim, CamType.NORMAL)

    def generate_fisheye_intrinsics(self, objpoints, imgpoints, imgs_dim):
        """
        Largely from https://github.com/mesutpiskin/opencv-fisheye-undistortion/blob/master/src/python/fisheye_calibration_undistortion.py
        """

        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        objpoints = np.expand_dims(np.asarray(objpoints), -2) # fix suggested at https://github.com/opencv/opencv/issues/9150

        # remove ill-conditioned images (https://stackoverflow.com/questions/49038464/opencv-calibrate-fisheye-lens-error-ill-conditioned-matrix)
        print("Removing ill-conditioned images...")
        original_num_images = len(objpoints)
        while True:
            assert len(objpoints) > 0, "There are no valid images from which to calibrate."
            N_OK = len(objpoints)
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            try:
                rms, _, _, _, _ = cv2.fisheye.calibrate(
                    objpoints,
                    imgpoints, 
                    imgs_dim,
                    K,
                    D,
                    rvecs,
                    tvecs,
                    calibration_flags,
                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                )
                print(f"Found a calibration with RMS of {rms} based on {len(objpoints)} well-conditioned images, pruned down from the original {original_num_images} images.")
                break
            except cv2.error as err:
                try:
                    print(str(err))
                    idx = int(str(err).split('array ')[1][0])  # Parse index of invalid image from error message
                    objpoints = np.delete(objpoints, idx, axis=0)
                    imgpoints = np.delete(imgpoints, idx, axis=0)
                    print("Removed ill-conditioned image {} from the data.  Trying again...".format(idx))
                except IndexError:
                    raise err
                
        return SingleCameraParameters(K, D, imgs_dim, CamType.FISHEYE)
     
class StereoCameraParams():
    def __init__(self, cam1_params, cam2_params, R, T, cam1_rectification = None, cam2_rectification = None, cam1_projection = None, cam2_projection = None, disparityToDepthMap = None, cam1ROI = None, cam2ROI = None):
        # cam1 == left view, cam2 == right view
        self.cam1_params = cam1_params
        self.cam2_params = cam2_params
        self.R = R
        self.T = T
        self.cam1_rectification = cam1_rectification
        self.cam2_rectification = cam2_rectification
        self.cam1_projection = cam1_projection
        self.cam2_projection = cam2_projection
        self.disparityToDepthMap = disparityToDepthMap
        self.cam1ROI = cam1ROI
        self.cam2ROI = cam2ROI

class StereoCameraCalibrator(Calibrator):
    def __init__(self, cam1_frames_path, cam2_frames_path, checkerboard_dims = (6, 9), square_size = 1, criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
                 cam_type = CamType.NORMAL):
        super().__init__()
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        if not os.path.exists(cam1_frames_path):
            raise ValueError(f"Frames path {cam1_frames_path} passed to StereoCameraCalibrator does not exist")
        if not os.path.exists(cam2_frames_path):
            raise ValueError(f"Frames path {cam2_frames_path} passed to StereoCameraCalibrator does not exist")
        self.cam1_frames_path = cam1_frames_path
        self.cam2_frames_path = cam2_frames_path
        self.checkerboard_dims = checkerboard_dims
        self.square_size = square_size
        self.criteria = criteria
        self.cam_type = cam_type

    def calibrate(self):
        print("Calibrating stereo camera...")
        cam1_params = SingleCameraCalibrator(self.cam1_frames_path, self.checkerboard_dims, self.square_size, self.criteria, self.cam_type).calibrate()
        cam2_params = SingleCameraCalibrator(self.cam2_frames_path, self.checkerboard_dims, self.square_size, self.criteria, self.cam_type).calibrate()
        
        objpoints_left, imgpoints_left, dim_left = self.get_calibration_points(self.cam1_frames_path)
        objpoints_right, imgpoints_right, dim_right = self.get_calibration_points(self.cam2_frames_path)
        assert dim_left == dim_right, "Frame dimensions must match."
        objpoints = np.array([objpoints_left[i] for i in range(len(objpoints_left)) if objpoints_left[i] is not None and objpoints_right[i] is not None])
        imgpoints_left = np.array([imgpoints_left[i] for i in range(len(imgpoints_left)) if objpoints_left[i] is not None and objpoints_right[i] is not None])
        imgpoints_right = np.array([imgpoints_right[i] for i in range(len(imgpoints_right)) if objpoints_left[i] is not None and objpoints_right[i] is not None])

        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        ret, CM1, CD1, CM2, CD2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, cam1_params.K, cam1_params.D, cam2_params.K, cam2_params.D, 
                                                                dim_right, criteria = self.criteria, flags = stereocalibration_flags)
        
        assert np.array_equal(np.array(CM1), np.array(cam1_params.K)), "Calibrated camera matrix 1 does not match input camera matrix 1."
        assert np.array_equal(np.array(CD1), np.array(cam1_params.D)), "Calibrated distortion coefficients 1 do not match input distortion coefficients 1."
        assert np.array_equal(np.array(CM2), np.array(cam2_params.K)), "Calibrated camera matrix 2 does not match input camera matrix 2."
        assert np.array_equal(np.array(CD2), np.array(cam2_params.D)), "Calibrated distortion coefficients 2 do not match input distortion coefficients 2."

        print("Reprojection error (pixels):", ret) # below 1 is good

        # Get rectification parameters
        (leftRectification, rightRectification, leftProjection, rightProjection,
        disparityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                cam1_params.K, cam1_params.D,
                cam2_params.K, cam2_params.D,
                cam1_params.cal_dims, R, T,
                None, None, None, None, None,
                cv2.CALIB_ZERO_DISPARITY, alpha=-1)

        return StereoCameraParams(cam1_params, cam2_params, R, T, leftRectification, rightRectification, leftProjection, rightProjection, disparityToDepthMap, leftROI, rightROI)


def undistort(img, camParams):
    """
    Undistort the image using the camera matrix and distortion coefficients
    """
    if camParams.camType == CamType.NORMAL:
        undistorted_img = cv2.undistort(img, camParams.K, camParams.D)
    else:
        # v2 (https://github.com/mesutpiskin/opencv-fisheye-undistortion/blob/master/src/python/fisheye_calibration_undistortion.py)
        DIM = camParams.cal_dims
        balance=1
        dim2=None
        dim3=None
        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if not dim2:
            dim2 = dim1
        if not dim3:
            dim3 = dim1
        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0 # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, camParams.D , dim2, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, camParams.D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        display = np.hstack((img, undistorted_img))
        cv2.imshow('V2: Original + Undistorted', display)
        cv2.waitKey(0)
    return undistorted_img


# stereo_cam_params = StereoCameraCalibrator("stereo_calibration_input/jul21/images/LEFT", "stereo_calibration_input/jul21/images/RIGHT", square_size = 29).calibrate()
# with open("params/stereo_params_jul21.pkl", "wb") as f:
#     pickle.dump(stereo_cam_params, f)