cmake_minimum_required(VERSION 2.8.3)
project(poli)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++0x")
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

include_directories( include
					 ${Boost_INCLUDE_DIRS}
					 ${catkin_INCLUDE_DIRS}
)

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

# add_executable( critic_node nodes/critic_node.cpp )
# add_dependencies( critic_node advant_gencfg ${advant_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} )
# target_link_libraries( critic_node advant ${catkin_LIBRARIES} )

# add_executable( monte_carlo_value_node nodes/monte_carlo_value_node.cpp )
# add_dependencies( monte_carlo_value_node ${PROJECT_NAME}_gencfg )
# target_link_libraries( monte_carlo_value_node
#                       ${catkin_LIBRARIES}
#                       ${Boost_LIBRARIES}
#                       valu )

# add_executable( value_learner_node nodes/value_learner_node.cpp )
# add_dependencies( value_learner_node ${PROJECT_NAME}_gencfg )
# target_link_libraries( value_learner_node
#                       ${catkin_LIBRARIES}
#                       ${Boost_LIBRARIES}
#                       valu )

# add_executable( approximate_value_node nodes/value_approximator_node.cpp )
# add_dependencies( approximate_value_node ${PROJECT_NAME}_gencfg )
# target_link_libraries( approximate_value_node
#                       ${catkin_LIBRARIES}
#                       ${Boost_LIBRARIES}
#                       valu )

# add_executable( srs_sampler_node nodes/srs_sampler_node.cpp )
# add_dependencies( srs_sampler_node ${PROJECT_NAME}_gencfg )
# target_link_libraries( srs_sampler_node
#                        ${catkin_LIBRARIES}
#                        ${Boost_LIBRARIES}
#                        valu )

# add_executable( td_error_node nodes/td_error_node.cpp )
# add_dependencies( td_error_node ${PROJECT_NAME}_gencfg )
# target_link_libraries( td_error_node
                        #                       ${catkin_LIBRARIES}
                        #                       ${Boost_LIBRARIES}
                        #                       valu )

# add_executable( continuous_policy_learner_node nodes/continuous_policy_learner_node.cpp )
# add_dependencies( continuous_policy_learner_node ${PROJECT_NAME}_gencfg )
# target_link_libraries( continuous_policy_learner_node
# 						${catkin_LIBRARIES}
# 						${Boost_LIBRARIES}
# 						relearn )
						

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
