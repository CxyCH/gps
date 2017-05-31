""" Hyperparameters for PR2 trajectory optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.ros.agent_turtlebot import AgentTurtlebot
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_obstacles import CostObstacle
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.proto.gps_pb2 import MOBILE_POSITION, MOBILE_ORIENTATION, \
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR, ACTION, \
				POSITION_NEAREST_OBSTACLE
from gps.utility.general_utils import get_ee_points
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    MOBILE_POSITION: 3,
    MOBILE_ORIENTATION: 4,
    MOBILE_VELOCITIES_LINEAR: 3,
    MOBILE_VELOCITIES_ANGULAR: 3,
    ACTION: 3
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/turtlebot_example/'

# NOTE: This is odom pose.
# Default is in one_stacle.world

odom_pose = [1.5, 8.3, 0.]
x0s = []
reset_conditions = []
# NOTE: This is map pose (The order of quaternion also different).
map_state = [np.array([3.5, 8.3, 0.,	# Position x, y, z
								1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
								0., 0., 0.,			# Linear Velocities
								0., 0., 0.])]		# Angular Velocities
common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
    'use_mpc': True,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

for i in xrange(common['conditions']):
		idx_pos = SENSOR_DIMS[MOBILE_POSITION]
		idx_ori = SENSOR_DIMS[MOBILE_ORIENTATION]
		
		state = np.zeros(map_state[i].size)
		state[:idx_pos] = map_state[i][:idx_pos] - odom_pose
		
		# odom state is independent to map state
		state[idx_pos:idx_pos+idx_ori] = [0., 0., 0., 1.]
		
		x0s.append(state)
		reset_conditions.append(map_state[i])

agent = {
    'type': AgentTurtlebot,
    'dt': 0.05,
    'conditions': common['conditions'],
    'T': 100,
    'x0': x0s,
    'use_mpc': common['use_mpc'],
    'M': 5,
    'reset_conditions': reset_conditions,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [MOBILE_POSITION, MOBILE_ORIENTATION, \
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR],
    'obs_include': [],
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'use_mpc': common['use_mpc'],
    'iterations': 10,
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 5.0,
    'pos_gains': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}

algorithm['init_mpc'] = {
    'type': init_pd,
    'init_var': 5.0,
    'pos_gains': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['M'],
}

action_cost = {
    'type': CostAction,
    'wu': np.ones(SENSOR_DIMS[ACTION])*5e-3
}

state_cost = {
    'type': CostState,
    'data_types' : {
        MOBILE_ORIENTATION: {
            'wp': np.ones(SENSOR_DIMS[MOBILE_ORIENTATION])*100.,
            'target_state': np.array([0., 0., 0., 1.]),
        },
        MOBILE_VELOCITIES_LINEAR: {
            'wp': np.ones(SENSOR_DIMS[MOBILE_VELOCITIES_LINEAR])*100.0,
            'target_state': np.array([1.0, 0., 0.]),
        },
        MOBILE_VELOCITIES_ANGULAR: {
            'wp': np.ones(SENSOR_DIMS[MOBILE_VELOCITIES_ANGULAR])*2.5,
            'target_state': np.array([0., 0., 0.]),
        },
    },
}

obstacle_cost = {
    'type': CostObstacle,
    'obstacle_type' : POSITION_NEAREST_OBSTACLE,
    'position_type': MOBILE_POSITION,
		'wp': np.ones(SENSOR_DIMS[MOBILE_POSITION]),
		'd_safe': 0.4
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost, obstacle_cost],
    'weights': [1.0, 1.0, 25.0],
}


algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': 0,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': 5,
}

common['info'] = generate_experiment_info(config)
