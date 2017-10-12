cmake_minimum_required(VERSION 2.8.3)
project(adel)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++0x")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED 
	COMPONENTS      rospy 
	                message_generation
			        std_msgs
			        argus_utils
)

add_message_files(
	FILES           CheckpointNotification.msg
)

catkin_python_setup()

generate_messages(
	DEPENDENCIES    std_msgs
)

catkin_package(
	CATKIN_DEPENDS 	rospy
	                std_msgs
					message_runtime
					argus_utils
	
	DEPENDS         Boost 
					system_lib
)
