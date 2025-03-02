
load("stereo_params/single_sensor_v2_stereo_cam_params_3.mat")

left_img = imread("../box_tests/single_sensor_v2/take3/bottom/LEFT/27.png");
right_img = imread("../box_tests/single_sensor_v2/take3/bottom/RIGHT/27.png");

left_undistorted = undistortImage(left_img, stereoParams.CameraParameters1);
right_undistorted = undistortImage(right_img, stereoParams.CameraParameters2);


% %undistort each
% figure
% tiledlayout(2,2)
% 
% ax1 = nexttile;
% imshow(left_img)
% title("Left")
% 
% ax2 = nexttile;
% imshow(left_undistorted)
% title("Left Undistorted")
% 
% ax3 = nexttile;
% imshow(right_img)
% title("Right")
% 
% ax4 = nexttile;
% imshow(right_undistorted)
% title("Right Undistorted")
% 
% clf;


%select points in each
imshow(left_undistorted);
title("Left Undistorted")
hold on;
[left_x,left_y,button] = ginput(4);
for index = 1:4
    plot(left_x(index), left_y(index), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;

clf;
imshow(right_undistorted);
title("Right Undistorted")
hold on;
[right_x,right_y,button] = ginput(4);
for index = 1:4
    plot(right_x(index), right_y(index), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;

clf;


% triangulate1
figure
hold on;
points3d = [];
for i = 1:4
    point3d = triangulate([left_x(i), left_y(i)], [right_x(i), right_y(i)], stereoParams);
    disp([left_x(i), left_y(i)]);
    disp([right_x(i), right_y(i)]);
    disp(point3d)
    scatter3(point3d(1), point3d(2), point3d(3));
end
view(40,35)
hold off;
