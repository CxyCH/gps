# This message is published to the C++ controller to start
# a mobile robot
int32 id  # ID must be echoed back in SampleResult
ControllerParams controller

# Trial information
int32 T  # Trajectory length
float64 frequency  # Controller frequency
int8[] state_datatypes  # Which data types to include in state
int8[] obs_datatypes # Which data types to include in observation
float64[] position_tgt # A 3x1 array containing target position
float64[] quaternion_tgt # A 4x1 array containing target orientation
