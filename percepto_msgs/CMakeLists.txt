cmake_minimum_required(VERSION 2.8.3)
project(percepto_msgs)

find_package(catkin REQUIRED COMPONENTS
                    message_generation
                    geometry_msgs
                    std_msgs
                    sensor_msgs
)

## Generate service files
add_message_files( FILES
                   ContinuousAction.msg
                   DiscreteAction.msg
                   SARTuple.msg
                   SRSTuple.msg
)

add_service_files( FILES
                   GetCritique.srv
                   GetParameters.srv
                   RunOptimization.srv
                   SetParameters.srv )

generate_messages( DEPENDENCIES 
                   std_msgs
)

catkin_package( CATKIN_DEPENDS message_runtime 
                               std_msgs 
)
