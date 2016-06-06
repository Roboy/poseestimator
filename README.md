### pose_estimation ###
This is a model based pose_estimation, based on the paper "A Geometric Approach to Joint 2D Region-Based Segmentation and 3D Pose Estimation Using a 3D Shape Prior".
You can find it in the paper folder.

### dependencies ###
eigen3, sfml, glew, imagemagick, assimp, opengl, cuda, opencv, sdformat, pcl, boost

### hardware requirements ###
you will nee a cuda capable graphic card (this code was tested with a NVidia [GeForce GTX 960](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-960/specifications)) 

### build ###
you will need to edit the root path for the models in the main.cpp
```
#!bash
cd path/to/pose_estimation
mkdir build
cd build
cmake ..
make -j4
```
### run ###
```
#!bash
cd path/to/pose_estimation/bin
./poseestimator mesh_model lambda_trans lambda_rot
```
The mesh_model is the model you want to use (can be .dae or .sdf file). The folder will be searched for this model and the first instance will be used. The two lambda parameters define the initial learning for translation and rotation. Example:
```
#!bash
./poseestimator sphere.dae 0.00000001 0.0000001
```
