cmake_minimum_required(VERSION 2.8.3)
project(bandito)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

catkin_package(
               CATKIN_DEPENDS rospy
                              percepto_msgs

               DEPENDS  Boost 
                        system_lib
               )

catkin_python_setup()
