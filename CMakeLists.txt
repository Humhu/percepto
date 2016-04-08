project( percepto )

cmake_minimum_required( VERSION 2.8 )

if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()
message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")

add_definitions( -Wall -std=c++0x )

set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib" CACHE PATH
    "Target directory for all libraries.")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin" CACHE PATH
    "Target directory for all executables.")
    
set( percepto_VERSION_MAJOR 0 )
set( percepto_VERSION_MINOR 1 )
set( percepto_VERSION ${percepto_VERSION_MAJOR}.${percepto_VERSION_MINOR} )

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" )

set( INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include" )
set( LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib" )
set( BINARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin" )

## Finding 3rd party packages
find_package( Boost REQUIRED COMPONENTS random )
include_directories( ${Boost_INCLUDE_DIRS} )
list( APPEND percepto_LIBRARIES ${Boost_LIBRARIES} )

find_package( Eigen REQUIRED )
message( STATUS "Eigen found at: ${EIGEN_INCLUDE_DIRS}" )
include_directories( ${EIGEN_INCLUDE_DIRS} )

find_package( GSL REQUIRED )
message( STATUS "GSL found at: ${GSL_INCLUDE_DIRS}" )
include_directories( ${GSL_INCLUDE_DIRS} )
list( APPEND percepto_LIBRARIES ${GSL_LIBRARIES} )

include_directories( include )

add_library( percepto SHARED
             src/AdamStepper.cpp
             src/Backprop.cpp
             src/HingeActivation.cpp
             src/PerceptoUtils.cpp
             src/NullActivation.cpp
             src/SigmoidActivation.cpp
             src/SimpleConvergence.cpp )
target_link_libraries( percepto ${percepto_LIBRARIES})

add_executable( test_net tests/TestNet.cpp )
target_link_libraries( test_net percepto ${percepto_LIBRARIES} )
