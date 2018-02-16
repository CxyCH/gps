""" Default configuration and hyperparameter values for costs. """
import numpy as np

from gps.algorithm.cost.cost_utils import RAMP_CONSTANT, evallogl2term
from gps.proto.gps_pb2 import PEDSIM_AGENT, END_EFFECTOR_POINTS, POSITION_NEAREST_OBSTACLE

# CostFK
COST_FK = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'wp': None,  # State weights - must be set.
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'env_target': True,  # TODO - This isn't used.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-5,
    'target_end_effector': None,  # Target end-effector position.
    'evalnorm': evallogl2term,
}


# CostState
COST_STATE = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-2,
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'data_types': {
        'JointAngle': {
            'target_state': None,  # Target state - must be set.
            'wp': None,  # State weights - must be set.
        },
    },
}

# CostObstacle
COST_OBSTACLE = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-2,
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'wp': None,
    'obstacle_type': POSITION_NEAREST_OBSTACLE,
    'position_type': END_EFFECTOR_POINTS,
    'd_safe': 0.4,
}

# CostObstacle
COST_PEDESTRIAN = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-2,
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'wp': None,
    'pedestrian_type': PEDSIM_AGENT,
    'max_agents': 10,
    'd_safe': 0.4,
}

# CostBinaryRegion
COST_BINARY_REGION = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-2,
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'data_types': {
        'JointAngle': {
            'target_state': None,  # Target state - must be set.
            'wp': None,  # State weights - must be set.
        },
    },
    'max_distance': 0.1,
    'outside_cost': 1.0,
    'inside_cost': 0.0,
}

# CostSum
COST_SUM = {
    'costs': [],  # A list of hyperparam dictionaries for each cost.
    'weights': [],  # Weight multipliers for each cost.
}


# CostAction
COST_ACTION = {
    'wu': np.array([]),  # Torque penalties, must be 1 x dU numpy array.
}


# CostLinWP
COST_LIN_WP = {
    'waypoint_time': np.array([1.0]),
    'ramp_option': RAMP_CONSTANT,
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-5,
    'logalpha': 1e-5,
    'log': 0.0,
}
