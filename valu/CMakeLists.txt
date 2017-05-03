cmake_minimum_required(VERSION 2.8.3)
project(valu)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED 
             COMPONENTS	roscpp
             argus_utils
             modprop
             optim
             broadcast
             percepto_msgs )

#include( ExternalProject )
#ExternalProject_Add( lwpr_external
#  GIT_REPOSITORY git@github.com:Humhu/lwpr.git
#  BUILD_COMMAND cmake --build .
#  INSTALL_COMMAND make install
#  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CATKIN_DEVEL_PREFIX}
#)
#set( lwpr_INCLUDE_DIRS ${CATKIN_DEVEL_PREFIX}/include )
#set( lwpr_LIBRARIES ${CATKIN_DEVEL_PREFIX}/lib/liblwpr.so )

# Use Boost for most utilities, threading
find_package(Boost REQUIRED COMPONENTS random)

catkin_package(
               INCLUDE_DIRS 	include

               LIBRARIES		valu

               CATKIN_DEPENDS 	roscpp 
               argus_utils
               modprop
               optim
               broadcast
               percepto_msgs

               DEPENDS  Boost 
                        system_lib
               )

add_library( valu
            src/ApproximateValue.cpp
            src/ApproximateValueLearner.cpp
            src/ApproximateValueManager.cpp
            src/MonteCarloValue.cpp
            src/RewardInterpolater.cpp
            src/SRSSampler.cpp
            src/ValuCommon.cpp
            src/ValueInfoManager.cpp
            src/ValueFunctionModules.cpp
            src/ValueLearningProblem.cpp
            src/ValueResidualModules.cpp )
add_dependencies( valu lwpr_external valu_gencfg ${valu_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} )
target_link_libraries( valu
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
                      valu )

add_executable( value_learner_node nodes/value_learner_node.cpp )
add_dependencies( value_learner_node ${PROJECT_NAME}_gencfg )
target_link_libraries( value_learner_node
                      ${catkin_LIBRARIES}
                      ${Boost_LIBRARIES}
                      valu )

# add_executable( approximate_value_node nodes/value_approximator_node.cpp )
# add_dependencies( approximate_value_node ${PROJECT_NAME}_gencfg )
# target_link_libraries( approximate_value_node
#                       ${catkin_LIBRARIES}
#                       ${Boost_LIBRARIES}
#                       valu )

add_executable( srs_sampler_node nodes/srs_sampler_node.cpp )
add_dependencies( srs_sampler_node ${PROJECT_NAME}_gencfg )
target_link_libraries( srs_sampler_node
                       ${catkin_LIBRARIES}
                       ${Boost_LIBRARIES}
                       valu )

# add_executable( td_error_node nodes/td_error_node.cpp )
# add_dependencies( td_error_node ${PROJECT_NAME}_gencfg )
# target_link_libraries( td_error_node
                        #                       ${catkin_LIBRARIES}
                        #                       ${Boost_LIBRARIES}
                        #                       valu )

install( TARGETS valu 
        monte_carlo_value_node
        value_learner_node
        # approximate_value_node
        # td_error_node
        srs_sampler_node
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

## Mark executables and/or libraries for installation
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h *.hpp"
        )
