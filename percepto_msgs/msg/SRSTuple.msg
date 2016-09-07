# A state-reward-state (SRS) tuple

# The time at which state and reward are received
std_msgs/Header header

# State at the header time
float64[] state

# Reward received at the header time
float64 reward

# Time for the next state
time next_time

# State at the next time
float64[] next_state