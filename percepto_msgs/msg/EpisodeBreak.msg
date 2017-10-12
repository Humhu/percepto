# EPISODEBREAK
#
# Indicates the end of an episode or the beginning of a new one
# Used for grouping streams of data to use in reinforcement learning
#
# Fields
# ------
# time  : ROS time at which the break occurs
# start : True if the break is a start event, False if it is an end event
# cause : Optional string detailing the event

time time
bool start
string cause