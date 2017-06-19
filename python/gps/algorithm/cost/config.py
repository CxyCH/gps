""" Default configuration and hyperparameter values for costs. """
import numpy as np
import os

from gps.algorithm.cost.cost_utils import RAMP_CONSTANT, evallogl2term


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
    'obstacle_type': None,
    'position_type': None,
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


# config options for any cost function learned through IOC
IOC_CONFIG = {  # TODO - maybe copy this from policy_opt/config
    'ioc_loss': 'ICML',  # Type of loss to use (ICML, XENTGAN, IOCGAN, MPF)
    'dO':0, # Number of features
    'T': 0, # the time horizon
    'iterations': 5000,  # Number of training iterations.
    'demo_batch_size': 5,  # Number of demos per mini-batch.
    'sample_batch_size': 5,  # Number of samples per mini-batch.
    'lr': 0.001,  # Base learning rate (by default it's fixed).
    'lr_policy': 'fixed',  # Learning rate policy.
    'solver_type': 'Adam',  # solver type (e.g. 'SGD', 'Adam')
    'momentum': 0.9,  # Momentum.
    'weight_decay': 0.0,  # Weight decay.
    'random_seed': 1,  # Random seed.
    # Set gpu usage.
    'use_gpu': 1,  # Whether or not to use the GPU for training.
    'gpu_id': 0,
    'smooth_reg_weight': 0.0,
    'mono_reg_weight': 100,
    'gp_reg_weight': 0.0, # Make it zero for now.
    'learn_wu': False,  # Learn multiplier on torque penalty, in addition to wu.
    'wu': np.array([]), # Torque penalties, must be 1 x dU numpy array.
    'weights_file_prefix': '',
    'use_jacobian': False,
    'network_arch_params': {},  # includes info to construct model
}

checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '..', 'cost/tf_checkpoint/cost_checkpoint.ckpt'))
# CostIOCTF
COST_IOC_TF = {
    # Other hyperparameters.
    'checkpoint_prefix': checkpoint_path,
    'approximate_lxx': True,
}
COST_IOC_TF.update(IOC_CONFIG)

# CostIOCTF + Vision
COST_IOC_VISION_TF = {
    # Other hyperparameters.
    'fc_only_iters': None,
    'approximate_lxx': True,
}
COST_IOC_VISION_TF.update(COST_IOC_TF)


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

