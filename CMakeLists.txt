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
list( APPEND percepto_LIBRARIES ${Boost_LIBRARIES} )
list( APPEND percepto_INCLUDES ${Boost_INCLUDE_DIRS} )

# Find Eigen linear algebra library
find_package( Eigen REQUIRED )
message( STATUS "Eigen found at: ${EIGEN_INCLUDE_DIRS}" )
list( APPEND percepto_INCLUDES ${EIGEN_INCLUDE_DIRS} )

# Find GNU Scientific Library
find_package( GSL REQUIRED )
message( STATUS "GSL found at: ${GSL_INCLUDE_DIRS}" )
list( APPEND percepto_LIBRARIES ${GSL_LIBRARIES} )
list( APPEND percepto_INCLUDES ${GSL_INCLUDE_DIRS} )

# Find LibLBFGS
find_path(LIBLBFGS_INCLUDE_DIR lbfgs.h)
find_library(LIBLBFGS_LIBRARY NAMES lbfgs liblbfgs )
message( STATUS "liblbfgs found at: ${LIBLBFGS_INCLUDE_DIR}" ) 
message( STATUS "liblbfgs lib at: ${LIBLBFGS_LIBRARY}" )
list( APPEND percepto_LIBRARIES ${LIBLBFGS_LIBRARY})
list( APPEND percepto_INCLUDES ${LIBLBFGS_INCLUDE_DIR} )

# Find NLOpt
find_path(NLOPT_INCLUDE_DIR nlopt.hpp)
find_library(NLOPT_LIBRARY NAMES nlopt )
message( STATUS "nlopt found at: ${NLOPT_INCLUDE_DIR}" )
message( STATUS "nlopt lib at: ${NLOPT_LIBRARY}" )
list( APPEND percepto_LIBRARIES ${NLOPT_LIBRARY})
list( APPEND percepto_INCLUDES ${NLOPT_INCLUDE_DIR} )

include_directories( ${percepto_INCLUDES} )

# Set install directories
set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(BINARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin")

# Header-only percepto library target
file(GLOB HEADERS "${PROJECT_SOURCE_DIR}/include/percepto/*.h"
    "${PROJECT_SOURCE_DIR}/include/percepto/*.hpp")
include_directories( include )

install( DIRECTORY "include/percepto"
         DESTINATION "${INCLUDE_INSTALL_DIR}"
         COMPONENT dev )

# Test binaries
add_executable( test_regression tests/RegressionTest.cpp )
target_link_libraries( test_regression ${percepto_LIBRARIES} )

add_executable( test_derivative tests/DerivativeTest.cpp )
target_link_libraries( test_derivative ${percepto_LIBRARIES} )

add_executable( test_optimization tests/OptimizationTest.cpp )
target_link_libraries( test_optimization ${percepto_LIBRARIES} )

add_executable( test_net tests/NetworkTest.cpp )
target_link_libraries( test_net ${percepto_LIBRARIES} )

install( TARGETS test_regression test_derivative test_optimization 
                 test_net
    RUNTIME DESTINATION bin )

configure_file( perceptoConfig.cmake.in
    "${PROJECT_BINARY_DIR}/perceptoConfig.cmake" @ONLY )

configure_file( perceptoConfigVersion.cmake.in
    "${PROJECT_BINARY_DIR}/perceptoConfigVersion.cmake" @ONLY )

install( FILES
    "${PROJECT_BINARY_DIR}/perceptoConfig.cmake"
    "${PROJECT_BINARY_DIR}/perceptoConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_PREFIX}" COMPONENT dev )
