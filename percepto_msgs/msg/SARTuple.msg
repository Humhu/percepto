# SARTuple
#
# A timestamped state-action-reward tuple. The timestamp corresponds
# to the state and action, as well as the accumulated reward until the
# next timestep.
# 
# Currently only supports continuous states and actions

# Contains source name and timestamp
std_msgs/Header header

float64[] state

float64[] action