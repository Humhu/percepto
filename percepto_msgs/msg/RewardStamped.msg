# RewardStamped
#
# A timestamped reward signal
#
# Fields
# ------
# header/stamp    : input/reward timestep
# header/frame_id : currently unused
# reward          : the instantaneous reward at this time

std_msgs/Header header
float64 reward
