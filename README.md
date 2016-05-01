### pose_estimation ###
This is a model based pose_estimation, based on the paper "A Geometric Approach to Joint 2D Region-Based Segmentation and 3D Pose Estimation Using a 3D Shape Prior".
You can find it in the paper folder.

### dependencies ###
eigen3, glfw3, glew, opencv

### build and run ###
```
#!bash
cd path/to/pose_estimation
cmake .
make
```
you will need to edit the paths to the shaders and the obj to be loaded. then:
```
#!bash
cd path/to/pose_estimation
./pose_estimation
```
