cmake_minimum_required(VERSION 2.8.3)
project(infitu)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS 
                    roscpp
                    message_generation
                    argus_utils
                    std_msgs
                    relearn
                    paraset
                    poli
                    dynamic_reconfigure
)

add_service_files( FILES
                   StartEvaluation.srv
                   StartTeardown.srv
                   SetRecording.srv
)

catkin_python_setup()

generate_messages()

# Use Boost for most utilities, threading
find_package(Boost REQUIRED COMPONENTS random)

# Use Eigen for matrices, linear algebra
find_package(Eigen3 REQUIRED)

catkin_package( INCLUDE_DIRS    include

                LIBRARIES       infitu

                CATKIN_DEPENDS  roscpp 
                                message_runtime
                                argus_utils
                                std_msgs
                                relearn
                                paraset
                                poli
                                dynamic_reconfigure

                DEPENDS         Boost
                                system_lib
                                Eigen3
)


include_directories(   include
                       ${EIGEN3_INCLUDE_DIR}
                       ${Boost_INCLUDE_DIRS}
                       ${catkin_INCLUDE_DIRS}
)

add_library(           infitu
                       src/ContinuousParamPolicy.cpp
                       #src/DiscretePolicy.cpp 
)
add_dependencies(      infitu 
                       infitu_gencfg 
                       ${catkin_EXPORTED_TARGETS} 
)
target_link_libraries( infitu ${catkin_LIBRARIES} )


install( TARGETS             infitu
         ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark executables and/or libraries for installation
install( DIRECTORY           include/${PROJECT_NAME}/
         DESTINATION         ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
         FILES_MATCHING PATTERN "*.h *.hpp"
)
