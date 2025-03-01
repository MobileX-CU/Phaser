load("stereo_params/dual_cam_stereo_cam_params_jun26.mat")

cam1 = plotCamera(AbsolutePose=stereoParams.PoseCamera2,Opacity=.1, AxesVisible=true, Color="blue");
initialRotation = [1  0  0;
                   0  1  0;
                   0 0  1];
initialTranslation = [0, 0, 0];
id_pose = rigidtform3d(initialRotation,initialTranslation);
hold on 
cam2 = plotCamera(AbsolutePose=id_pose, Opacity=.1, AxesVisible=true, Color="red");
grid on
axis equal

