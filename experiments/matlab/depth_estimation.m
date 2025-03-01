% save("stereo_cam_params_recent.mat", "stereoParams")

load("stereo_cam_params_recent.mat")
showExtrinsics(stereoParams);

videoFileLeft = "dual_jul23_left.mp4";
videoFileRight = "dual_jul23_right.mp4";

readerLeft = VideoReader(videoFileLeft);
readerRight = VideoReader(videoFileRight);
player = vision.VideoPlayer(Position=[20,200,740 560]);

frameLeft = readFrame(readerLeft);
frameRight = readFrame(readerRight);

[frameLeftRect, frameRightRect, reprojectionMatrix] = ...
    rectifyStereoImages(frameLeft, frameRight, stereoParams);

figure;
imshow(stereoAnaglyph(frameLeftRect, frameRightRect));
title("Rectified Video Frames");

frameLeftGray  = im2gray(frameLeftRect);
frameRightGray = im2gray(frameRightRect);

disparityMap = disparitySGM(frameLeftGray, frameRightGray);
figure;
imshow(disparityMap, [0, 64]);
title("Disparity Map");
colormap jet
colorbar

points3D = reconstructScene(disparityMap, reprojectionMatrix);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;
ptCloud = pointCloud(points3D, Color=frameLeftRect);

ptsOut = reshape(points3D, [], 3);
writematrix(ptsOut, 'points3d.csv');


% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-3, 3], [-3, 0], VerticalAxis="y", ...
    VerticalAxisDir="down");

% Visualize the point cloud
view(player3D, ptCloud);