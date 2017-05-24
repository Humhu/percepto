cmake_minimum_required(VERSION 2.8.3)
project(poli)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED 
	COMPONENTS		roscpp
					argus_utils
					broadcast
					modprop
					lookup
					percepto_msgs
)

# Use Boost for most utilities, threading
find_package(Boost REQUIRED COMPONENTS random)

catkin_package(
	# INCLUDE_DIRS 	include
						
	# LIBRARIES		poli

	CATKIN_DEPENDS 	roscpp 
					argus_utils
					broadcast
					modprop
					lookup
					percepto_msgs
		  
					
	DEPENDS 		Boost 
					system_lib
)

catkin_python_setup()

# add_library( poli
# 			src/ContinuousPolicy.cpp
# 			src/ContinuousPolicyManager.cpp
# 			src/ContinuousPolicyModules.cpp
# 			#src/DiscretePolicyManager.cpp
# 			#src/DiscretePolicyModules.cpp
# 			src/PoliCommon.cpp
# 			src/PolicyInfoManager.cpp )
# add_dependencies( poli ${poli_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} )
# target_link_libraries( poli
					#    ${catkin_LIBRARIES} )

include_directories( include
					 ${Boost_INCLUDE_DIRS}
					 ${catkin_INCLUDE_DIRS}
)

# install( TARGETS poli 
# 	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# 	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# 	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
install(DIRECTORY include/${PROJECT_NAME}/
	DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
	FILES_MATCHING PATTERN "*.h *.hpp"
)
