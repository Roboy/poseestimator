CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(poseestimator C CXX)

# RelWithDebInfo, Release or Debug
SET(CMAKE_BUILD_TYPE Debug FORCE)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -std=c++11 -D_MWAITXINTRIN_H_INCLUDED" )

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#==============================================================================
# Dependencies
#==============================================================================
# Eigen
find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

# SFML
pkg_check_modules( SFML REQUIRED sfml-graphics )

# glew
find_package(GLEW REQUIRED)

# CUDA
SET(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-7.5)
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -g -gencode=arch=compute_52,code=sm_52 --use_fast_math)

# OpenCV
FIND_PACKAGE(OpenCV 3.1.0 COMPONENTS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_calib3d REQUIRED)

INCLUDE_DIRECTORIES(include)

#==============================================================================
# Application
#==============================================================================
# preprocessor define: source directory
ADD_DEFINITIONS(-DTSDF_CUDA_SOURCE_DIR=${PROJECT_SOURCE_DIR})

CUDA_ADD_EXECUTABLE(poseestimator src/main.cpp src/poseestimator.cu)
TARGET_LINK_LIBRARIES(poseestimator ${OpenCV_LIBS} ${SFML_LIBRARIES} ${GLEW_LIBRARIES} -lpthread -lGL)

