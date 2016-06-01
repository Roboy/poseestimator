# - FindMathGL2.cmake
# This module can be used to find MathGL v.2.* and several of its optional components.
#
# You can specify one or more component as you call this find module.
# Possible components are: FLTK, GLUT, Qt, WX.
#
# The following variables will be defined for your use:
#
#  MATHGL2_FOUND           = MathGL v.2 and all specified components found
#  MATHGL2_INCLUDE_DIRS    = The MathGL v.2 include directories
#  MATHGL2_LIBRARIES       = The libraries to link against to use MathGL v.2
#                           and all specified components
#  MATHGL2_VERSION_STRING  = A human-readable version of the MathGL v.2 (e.g. 2.1)
#  MATHGL2_XXX_FOUND       = Component XXX found (replace XXX with uppercased
#                           component name -- for example, QT or FLTK)
#
# The minimum required version and needed components can be specified using
# the standard find_package()-syntax, here are some examples:
#  find_package(MathGL2 REQUIRED)				- v.2.* (no interfaces), required
#  find_package(MathGL2 2.1 REQUIRED Qt)		- v.2.1 + Qt interface, required
#  find_package(MathGL2 2.1 REQUIRED)			- v.2.1 (no interfaces), required
#  find_package(MathGL2 2.0 COMPONENTS Qt WX)	- v.2.0 + Qt and WX interfaces, optional
#  find_package(MathGL2 2.1)					- v.2.1 (no interfaces), optional
#
# Note, some cmake builds require to write "COMPONENTS" always, like
#  find_package(MathGL2 REQUIRED COMPONENTS Qt)	- v.2.* + Qt interface, required
#
# Typical usage could be something like this:
#   find_package(MathGL 2.1 GLUT REQUIRED)
#   include_directories(${MATHGL2_INCLUDE_DIRS})
#   add_executable(myexe main.cpp)
#   target_link_libraries(myexe ${MATHGL2_LIBRARIES})
#

#=============================================================================
# Copyright (c) 2011 Denis Pesotsky <denis@kde.ru>, 2014 Alexey Balakin <mathgl.abalakin@gmail.com>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file COPYING-CMAKE-MODULES for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================

FIND_PATH(MathGL_INCLUDE_DIRS NAMES mgl2/mgl.h
  PATHS
  /opt/local/include
  /usr/include
  /usr/local/include
)

FIND_LIBRARY(MathGL_LIB NAMES mgl
  PATHS
  /opt/local/lib
  /usr/local/lib
  /usr/lib
)
#FIND_LIBRARY(MathGL_QT_LIB NAMES mgl-qt
#  PATHS
#  /opt/local/lib
#  /usr/local/lib
#  /usr/lib
#)

FIND_LIBRARY(MathGL_GLUT_LIB NAMES mgl-glut
        PATHS
        /opt/local/lib
        /usr/local/lib
        /usr/lib
        )

SET(MathGL_LIBRARIES ${MathGL_LIB} ${MathGL_GLUT_LIB})

IF (MathGL_INCLUDE_DIRS AND MathGL_LIBRARIES)
  SET(MathGL_FOUND TRUE)
  MESSAGE(STATUS "MathGL found")
  MESSAGE(STATUS "MathGL Include dirs:" ${MathGL_INCLUDE_DIRS})
  MESSAGE(STATUS "MathGL Libraries:" ${MathGL_LIBRARIES})
ELSE (MathGL_INCLUDE_DIRS AND MathGL_LIBRARIES)
  MESSAGE(STATUS "MathGL was not found")
ENDIF(MathGL_INCLUDE_DIRS AND MathGL_LIBRARIES)
