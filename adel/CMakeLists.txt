cmake_minimum_required(VERSION 2.8.3)
project(adel)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++0x")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

set(catkin_required rospy)

find_package(catkin REQUIRED 
	COMPONENTS ${catkin_required}
)

# Use Boost for most utilities, threading
find_package(Boost REQUIRED COMPONENTS random)

catkin_package(
	CATKIN_DEPENDS 	${catkin_required}
					
	DEPENDS 		Boost 
					system_lib
)

catkin_python_setup()
