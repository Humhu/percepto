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
# Find Boost C++ Utility libraries
find_package( Boost REQUIRED COMPONENTS random )
include_directories( ${Boost_INCLUDE_DIRS} )
list( APPEND percepto_LIBRARIES ${Boost_LIBRARIES} )

# Find Eigen linear algebra library
find_package( Eigen REQUIRED )
message( STATUS "Eigen found at: ${EIGEN_INCLUDE_DIRS}" )
include_directories( ${EIGEN_INCLUDE_DIRS} )

# Find GNU Scientific Library
find_package( GSL REQUIRED )
message( STATUS "GSL found at: ${GSL_INCLUDE_DIRS}" )
include_directories( ${GSL_INCLUDE_DIRS} )
list( APPEND percepto_LIBRARIES ${GSL_LIBRARIES} )

# Find LibLBFGS
find_path(LIBLBFGS_INCLUDE_DIR lbfgs.h)
find_library(LIBLBFGS_LIBRARY NAMES lbfgs liblbfgs )
message( STATUS "liblbfgs found at: ${LIBLBFGS_INCLUDE_DIR}" ) 
message( STATUS "liblbfgs lib at: ${LIBLBFGS_LIBRARY}" )
include_directories(${LIBLBFGS_INCLUDE_DIR})
list( APPEND percepto_LIBRARIES ${LIBLBFGS_LIBRARY})

# Find NLOpt
find_path(NLOPT_INCLUDE_DIR nlopt.hpp)
find_library(NLOPT_LIBRARY NAMES nlopt )
message( STATUS "nlopt found at: ${NLOPT_INCLUDE_DIR}" )
message( STATUS "nlopt lib at: ${NLOPT_LIBRARY}" )
include_directories(${NLOPT_INCLUDE_DIR})
list( APPEND percepto_LIBRARIES ${NLOPT_LIBRARY})

include_directories( include )

#add_library( covreg SHARED
#             src/LEMCRTypes.cpp
#target_link_libraries( covreg ${percepto_LIBRARIES})

add_executable( test_regression tests/RegressionTest.cpp )
target_link_libraries( test_regression ${percepto_LIBRARIES} )

add_executable( test_derivative tests/DerivativeTest.cpp )
target_link_libraries( test_derivative ${percepto_LIBRARIES} )

add_executable( test_optimization tests/OptimizationTest.cpp )
target_link_libraries( test_optimization ${percepto_LIBRARIES} )

add_executable( test_net tests/NetworkTest.cpp )
target_link_libraries( test_net ${percepto_LIBRARIES} )
