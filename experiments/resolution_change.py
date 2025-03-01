import glob
import cv2 as cv
import os

def downsample_cal_frames(frames_path, output_path, factor):
    os.makedirs(f"{output_path}/LEFT", exist_ok = True)
    os.makedirs(f"{output_path}/RIGHT", exist_ok = True)
    frames_left = glob.glob(f"{frames_path}/LEFT/*.png")
    frames_right = glob.glob(f"{frames_path}/RIGHT/*.png")
    frames_left = sorted(frames_left, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    frames_right = sorted(frames_right, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for i, (frame_left, frame_right) in enumerate(zip(frames_left, frames_right)):
        # downsample each frame by factor
        img1 = cv2.imread(frame_left)
        img2 = cv2.imread(frame_right)
        img1 = cv2.resize(img1, (int(img1.shape[1] / factor), int(img1.shape[0] / factor)))
        img2 = cv2.resize(img2, (int(img2.shape[1] / factor), int(img2.shape[0] / factor)))
        print(img1.shape, img2.shape)
        cv2.imwrite(f"{output_path}/LEFT/{i}.png", img1)
        cv2.imwrite(f"{output_path}/RIGHT/{i}.png", img2)


def downsample_video(video_path, output_video_path, factor):
    vid_name = output_video_path.split('/')[-1]
    output_video_parent_dir = output_video_path.split(vid_name)[0]
    if not os.path.exists(output_video_parent_dir):
        os.makedirs(output_video_parent_dir)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(3) / factor), int(cap.get(4) / factor)))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (int(frame.shape[1] / factor), int(frame.shape[0] / factor)))
        out.write(frame)
    cap.release()
    out.release()

# downsample_cal_frames('dual_cal_jun23', f"resolution_exps/dual_cal_jun23_d19", 1.9)
# downsample_cal_frames('dual_cal_jun23', f"resolution_exps/dual_cal_jun23_d1", 1)

# downsample_video("box_tests/dual_sensor/full_resolution/box_test_top_left.mp4", "box_tests/dual_sensor/resolution_d15/box_test_top_d1_left.mp4", 1)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_top_right.mp4", "box_tests/dual_sensor/resolution_d15/box_test_top_d1_right.mp4", 1)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_bottom_left.mp4", "box_tests/dual_sensor/resolution_d15/box_test_bottom_d1_left.mp4", 1)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_bottom_right.mp4", "box_tests/dual_sensor/resolution_d15/box_test_bottom_d1_right.mp4", 1)


# downsample_video("box_tests/dual_sensor/full_resolution/box_test_top_left.mp4", "box_tests/dual_sensor/resolution_d15/box_test_top_d15_left.mp4", 1.5)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_top_right.mp4", "box_tests/dual_sensor/resolution_d15/box_test_top_d15_right.mp4", 1.5)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_bottom_left.mp4", "box_tests/dual_sensor/resolution_d15/box_test_bottom_d15_left.mp4", 1.5)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_bottom_right.mp4", "box_tests/dual_sensor/resolution_d15/box_test_bottom_d15_right.mp4", 1.5)

# # downsample videos x1.7
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_top_left.mp4", "box_tests/dual_sensor/resolution_d17/box_test_top_d17_left.mp4", 1.7)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_top_right.mp4", "box_tests/dual_sensor/resolution_d17/box_test_top_d17_right.mp4", 1.7)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_bottom_left.mp4", "box_tests/dual_sensor/resolution_d17/box_test_bottom_d17_left.mp4", 1.7)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_bottom_right.mp4", "box_tests/dual_sensor/resolution_d17/box_test_bottom_d17_right.mp4", 1.7)

# downsample_video("box_tests/dual_sensor/full_resolution/box_test_top_left.mp4", "box_tests/dual_sensor/resolution_d19/box_test_top_d19_left.mp4", 1.9)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_top_right.mp4", "box_tests/dual_sensor/resolution_d19/box_test_top_d19_right.mp4", 1.9)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_bottom_left.mp4", "box_tests/dual_sensor/resolution_d19/box_test_bottom_d19_left.mp4", 1.9)
# downsample_video("box_tests/dual_sensor/full_resolution/box_test_bottom_right.mp4", "box_tests/dual_sensor/resolution_d19/box_test_bottom_d19_right.mp4", 1.9)

