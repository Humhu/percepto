cmake_minimum_required(VERSION 2.8.3)
project(infitu)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++0x")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS 
                    roscpp
                    message_generation
                    argus_utils
                    std_msgs
                    paraset
                    poli
                    dynamic_reconfigure
					rosbag
)

add_service_files( FILES
                   StartEvaluation.srv
                   StartSetup.srv				   
                   StartTeardown.srv
                   SetRecording.srv
)

catkin_python_setup()

generate_messages()

# Use Boost for most utilities, threading
find_package(Boost REQUIRED COMPONENTS random)

catkin_package( 
				# INCLUDE_DIRS    include

                # LIBRARIES       infitu

                CATKIN_DEPENDS  roscpp 
                                message_runtime
                                argus_utils
                                std_msgs
                                paraset
                                poli
                                dynamic_reconfigure
								rosbag

                DEPENDS         Boost
                                system_lib
)


include_directories(   #include
                       ${Boost_INCLUDE_DIRS}
                       ${catkin_INCLUDE_DIRS}
)

# add_library(           infitu
                    #    src/ContinuousParamPolicy.cpp
                       #src/DiscretePolicy.cpp 
# )
# add_dependencies(      infitu 
                    #    ${catkin_EXPORTED_TARGETS} 
# )
# target_link_libraries( infitu ${catkin_LIBRARIES} )

add_executable( ape_reward_node nodes/rewards/ape_reward.cpp )
target_link_libraries( ape_reward_node ${catkin_LIBRARIES} )

add_executable( sse_reward_node nodes/rewards/sse_reward.cpp )
target_link_libraries( sse_reward_node ${catkin_LIBRARIES} )

add_executable( oll_reward_node nodes/rewards/obs_log_likelihood_reward.cpp )
target_link_libraries( oll_reward_node ${catkin_LIBRARIES} )

add_executable( reward_accumulator_node nodes/rewards/reward_accumulator.cpp )
target_link_libraries( reward_accumulator_node ${catkin_LIBRARIES} )

add_executable( bag_replayer nodes/bag_replayer.cpp )
target_link_libraries( bag_replayer ${catkin_LIBRARIES} )

add_executable( multi_bag_replayer nodes/multi_bag_replayer.cpp )
target_link_libraries( multi_bag_replayer ${catkin_LIBRARIES} )

# install( TARGETS             infitu
#          ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#          LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#          RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
install( DIRECTORY           include/${PROJECT_NAME}/
         DESTINATION         ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
         FILES_MATCHING PATTERN "*.h *.hpp"
)
