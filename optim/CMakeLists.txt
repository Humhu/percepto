cmake_minimum_required( VERSION 2.8.3 )
project( optim )

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules")

if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()
message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")

add_definitions( -Wall -std=c++0x )

find_package(catkin REQUIRED 
    COMPONENTS      roscpp
                    modprop
                    argus_utils
)

## Finding 3rd party packages
# Find Boost C++ Utility libraries
find_package( Boost REQUIRED COMPONENTS random )
list( APPEND optim_LIBRARIES ${Boost_LIBRARIES} )
list( APPEND optim_INCLUDES ${Boost_INCLUDE_DIRS} )

# Find Eigen linear algebra library
find_package( Eigen3 REQUIRED )
message( STATUS "Eigen found at: ${EIGEN3_INCLUDE_DIR}" )
list( APPEND optim_INCLUDES ${EIGEN3_INCLUDE_DIR} )

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" )

# Find GNU Scientific Library
#find_package( GSL REQUIRED )
#message( STATUS "GSL found at: ${GSL_INCLUDE_DIRS}" )
#list( APPEND optim_LIBRARIES ${GSL_LIBRARIES} )
#list( APPEND optim_INCLUDES ${GSL_INCLUDE_DIRS} )

# Find LibLBFGS
#find_path(LIBLBFGS_INCLUDE_DIR lbfgs.h)
#find_library(LIBLBFGS_LIBRARY NAMES lbfgs liblbfgs )
#message( STATUS "liblbfgs found at: ${LIBLBFGS_INCLUDE_DIR}" ) 
#message( STATUS "liblbfgs lib at: ${LIBLBFGS_LIBRARY}" )
#list( APPEND optim_LIBRARIES ${LIBLBFGS_LIBRARY})
#list( APPEND optim_INCLUDES ${LIBLBFGS_INCLUDE_DIR} )

# Find NLOpt
#find_path(NLOPT_INCLUDE_DIR nlopt.hpp)
#find_library(NLOPT_LIBRARY NAMES nlopt )
#message( STATUS "nlopt found at: ${NLOPT_INCLUDE_DIR}" )
#message( STATUS "nlopt lib at: ${NLOPT_LIBRARY}" )
#list( APPEND optim_LIBRARIES ${NLOPT_LIBRARY} )
#list( APPEND optim_INCLUDES ${NLOPT_INCLUDE_DIR} )

catkin_package(
    INCLUDE_DIRS    include

    LIBRARIES       optim
                    
    CATKIN_DEPENDS  roscpp 
                    modprop
                    argus_utils
                    
    DEPENDS         Boost 
                    system_lib
                    Eigen3
)

catkin_python_setup()

include_directories( ${optim_INCLUDES} ${catkin_INCLUDE_DIRS} include )

add_library( optim 
             src/AdamSearchDirector.cpp
             src/ConstrainedBacktrackingSearchStepper.cpp
             src/ConstraintTerminationChecker.cpp
             src/BacktrackingSearchStepper.cpp
             src/FixedSearchStepper.cpp
             src/GradientSearchDirector.cpp
             src/GradientTerminationChecker.cpp
             src/IterationTerminationChecker.cpp
             src/L1ConstrainedSearchStepper.cpp
             src/L2ConstrainedSearchStepper.cpp
             src/ModularOptimizer.cpp
             src/NaturalSearchDirector.cpp
             src/OptimizationProfiler.cpp
             src/OptimizerParser.cpp
             src/RuntimeTerminationChecker.cpp )
target_link_libraries( optim ${catkin_LIBRARIES} ${optim_LIBRARIES})

add_executable( test_optimization tests/OptimizationTest.cpp )
target_link_libraries( test_optimization 
                       optim
                       ${modprop_LIBRARIES}
                       ${catkin_LIBRARIES}  )

add_executable( test_net tests/NetworkTest.cpp )
target_link_libraries( test_net 
                       optim
                       ${modprop_LIBRARIES}
                       ${catkin_LIBRARIES}  )

install( TARGETS optim
                 test_optimization
                 test_net
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)