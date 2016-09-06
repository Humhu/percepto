cmake_minimum_required(VERSION 2.8.3)
project(relearn)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED 
	COMPONENTS	roscpp
					argus_utils
          modprop
          poli
)

# Use Boost for most utilities, threading
find_package(Boost REQUIRED COMPONENTS random)

# Use Eigen for matrices, linear algebra
find_package(Eigen3 REQUIRED)

catkin_package(
	INCLUDE_DIRS 	include
						
	LIBRARIES		relearn

	CATKIN_DEPENDS 	roscpp 
					argus_utils
          modprop
          poli
					
	DEPENDS 		Boost 
					system_lib
)

add_library( relearn
            src/ApproximateValue.cpp
            src/ApproximateValueLearner.cpp
            src/ContinuousPolicyLearner.cpp
            src/DifferenceCritic.cpp
            src/MonteCarloValue.cpp
            src/PolicyLogGradientModules.cpp
            src/RewardInterpolater.cpp
            src/RunningAverageBaseline.cpp
            src/TDErrorCritic.cpp
            src/ValueFunctionModules.cpp
            src/ValueResidualModules.cpp )
add_dependencies( relearn relearn_gencfg ${relearn_EXPORTED_TARGETS} )
target_link_libraries( relearn
                       ${catkin_LIBRARIES} )

include_directories( include
${EIGEN3_INCLUDE_DIR}
					 ${Boost_INCLUDE_DIRS}
					 ${catkin_INCLUDE_DIRS}
)

add_executable( monte_carlo_value_node nodes/monte_carlo_value_node.cpp )
add_dependencies( monte_carlo_value_node ${PROJECT_NAME}_gencfg )
target_link_libraries( monte_carlo_value_node
                      ${catkin_LIBRARIES}
                      ${Boost_LIBRARIES}
                      relearn )

add_executable( continuous_policy_learner_node nodes/continuous_policy_learner_node.cpp )
add_dependencies( continuous_policy_learner_node ${PROJECT_NAME}_gencfg )
target_link_libraries( continuous_policy_learner_node
                      ${catkin_LIBRARIES}
                      ${Boost_LIBRARIES}
                      relearn )

add_executable( approximate_value_learner_node nodes/value_approximator_learner_node.cpp )
add_dependencies( approximate_value_learner_node ${PROJECT_NAME}_gencfg )
target_link_libraries( approximate_value_learner_node
                      ${catkin_LIBRARIES}
                      ${Boost_LIBRARIES}
                      relearn )

add_executable( approximate_value_node nodes/value_approximator_node.cpp )
add_dependencies( approximate_value_node ${PROJECT_NAME}_gencfg )
target_link_libraries( approximate_value_node
                      ${catkin_LIBRARIES}
                      ${Boost_LIBRARIES}
                      relearn )

# add_executable( td_error_node nodes/td_error_node.cpp )
# add_dependencies( td_error_node ${PROJECT_NAME}_gencfg )
# target_link_libraries( td_error_node
#                       ${catkin_LIBRARIES}
#                       ${Boost_LIBRARIES}
#                       relearn )

install( TARGETS relearn 
                 monte_carlo_value_node
                 continuous_policy_learner_node
                 approximate_value_learner_node
                 approximate_value_node
                 # td_error_node
	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark executables and/or libraries for installation
install(DIRECTORY include/${PROJECT_NAME}/
	DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
	FILES_MATCHING PATTERN "*.h *.hpp"
)
