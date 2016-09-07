cmake_minimum_required(VERSION 2.8.3)
project(advant)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED 
             COMPONENTS	roscpp
             argus_utils
             modprop
             broadcast
             valu
             percepto_msgs )

# Use Boost for most utilities, threading
find_package(Boost REQUIRED COMPONENTS random)

# Use Eigen for matrices, linear algebra
find_package(Eigen3 REQUIRED)

catkin_package(
               INCLUDE_DIRS 	include

               LIBRARIES		advant

               CATKIN_DEPENDS 	roscpp 
               argus_utils
               modprop
               broadcast
               percepto_msgs

               DEPENDS 		Boost 
               system_lib
               )

include_directories( include
                    ${EIGEN3_INCLUDE_DIR}
                    ${Boost_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
                    )

add_library( advant
            src/GeneralizedAdvantageCritic.cpp
            #src/DifferenceCritic.cpp
            #src/RunningAverageBaseline.cpp
            src/TDErrorCritic.cpp )
add_dependencies( advant advant_gencfg ${advant_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} )
target_link_libraries( advant
                      ${catkin_LIBRARIES} )

add_executable( critic_node nodes/critic_node.cpp )
add_dependencies( critic_node advant_gencfg ${advant_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} )
target_link_libraries( critic_node advant ${catkin_LIBRARIES} )

install( TARGETS advant 
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

## Mark executables and/or libraries for installation
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h *.hpp"
        )
