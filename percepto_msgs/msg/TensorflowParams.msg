# Message for tensorflow parameters

# Serialized and concatenated parameters
float32[] payload

# Lengths of each parameter in the payload, ordered
# length of payload = sum(sizes)
uint32[] sizes

# Names corresponding to each parameter
string[] names