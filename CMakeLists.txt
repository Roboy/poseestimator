CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(poseestimator C CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# RelWithDebInfo, Release or Debug
SET(CMAKE_BUILD_TYPE Debug FORCE)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -std=c++11 -DMAGICKCORE_QUANTUM_DEPTH=16 -DMAGICKCORE_HDRI_ENABLE=0" ) # -fext-numeric-literals

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

# image magick
find_package( ImageMagick COMPONENTS Magick++ MagickWand MagickCore REQUIRED)
include_directories(${ImageMagick_Magick++_INCLUDE_DIR})

# assimp
find_package(assimp REQUIRED)

# opengl
find_package(OpenGL REQUIRED)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-7.5)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DDEBUG -G -Xcompiler -g -gencode=arch=compute_52,code=sm_52 --use_fast_math)

# OpenCV
find_package(OpenCV 3.1.0 COMPONENTS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_videoio REQUIRED)

# sdformat
find_package(SDFormat REQUIRED)
include_directories(${SDFormat_INCLUDE_DIRS})

# pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# boost
find_package(Boost COMPONENTS filesystem REQUIRED)

# the following strips some vtk stuff, otherwise the nvcc compiler cannot deal with pcl
get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
        list(APPEND vtk_flags ${it})
    endif()
endforeach()

foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()

include_directories(include common_utilities/include)

message(STATUS ${MATHGL2_LIBRARIES} )

#==============================================================================
# Application
#==============================================================================
# preprocessor define: source directory
ADD_DEFINITIONS(-DTSDF_CUDA_SOURCE_DIR=${PROJECT_SOURCE_DIR})

# compile cuda stuff separately, mainly because of std++11, but also because of pcl which is incompatible
CUDA_ADD_LIBRARY(poseestimator src/poseestimator.cu STATIC )

ADD_EXECUTABLE(poseestimation src/main.cpp src/renderer.cpp src/mesh.cpp src/model.cpp )
TARGET_LINK_LIBRARIES(poseestimation ${SFML_LIBRARIES} ${GLEW_LIBRARIES} ${ASSIMP_LIBRARIES} ${OPENGL_LIBRARIES}
        ${ImageMagick_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${SDFormat_LIBRARIES} ${BOOST_LIBRARIES}
        poseestimator) #${MathGL_LIBRARIES}

