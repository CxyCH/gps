""" Hyperparameters for PR2 trajectory optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.ros.agent_mpepc_turtlebot import AgentMPEPCTurtlebot
from gps.algorithm.algorithm_pigps import AlgorithmPIGPS
from gps.algorithm.algorithm_pigps import AlgorithmMDGPS
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_obstacles import CostObstacle
from gps.algorithm.cost.cost_mpepc import CostMPEPC
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.proto.gps_pb2 import MOBILE_POSITION, MOBILE_ORIENTATION, \
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR, ACTION, \
				POSITION_NEAREST_OBSTACLE, POTENTIAL_SCORE, MOBILE_RANGE_SENSOR
from gps.utility.general_utils import get_ee_points
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    MOBILE_POSITION: 3,
    MOBILE_ORIENTATION: 4,
    MOBILE_VELOCITIES_LINEAR: 3,
    MOBILE_VELOCITIES_ANGULAR: 3,
    MOBILE_RANGE_SENSOR: 30,
    ACTION: 4
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/turtlebot_mpepc_hallway_pigps_example/'

# NOTE: This is odom pose.
# Default is in one_stacle.world
odom_pose = [3.5, 10.3, 0.]
#odom_pose = [3.5, 11.25, 0.] # Test hallway bend
x0s = []
reset_conditions = []

# NOTE: This is map pose (The order of quaternion also different).
map_state = [#np.array([5.5, 9.2, 0.,	# Position x, y, z
			#1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
			#0., 0., 0.,			# Linear Velocities
			#0., 0., 0.]),		# Angular Velocities
			np.array([3.5, 9.55, 0.,	# Position x, y, z
			1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
			0., 0., 0.,			# Linear Velocities
			0., 0., 0.]),		# Angular Velocities
			np.array([3.5, 10.3, 0.,	# Position x, y, z
			1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
			0., 0., 0.,			# Linear Velocities
			0., 0., 0.]),		# Angular Velocities
			np.array([3.5, 11.2, 0.,	# Position x, y, z
			1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
			0., 0., 0.,			# Linear Velocities
			0., 0., 0.]),		# Angular Velocities
			#np.array([5.5, 11.5, 0.,	# Position x, y, z
			#1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
			#0., 0., 0.,			# Linear Velocities
			#0., 0., 0.]),		# Angular Velocities
			]

goal_conditions = [#np.array([5.5, 9.2, 0.,	# Position x, y, z
			#1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
			#0., 0., 0.,			# Linear Velocities
			#0., 0., 0.]),		# Angular Velocities
			np.array([27.5, 12.3, 0.,	# Position x, y, z
			1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
			0., 0., 0.,			# Linear Velocities
			0., 0., 0.]),		# Angular Velocities
			np.array([27.5, 12.6, 0.,	# Position x, y, z
			1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
			0., 0., 0.,			# Linear Velocities
			0., 0., 0.]),		# Angular Velocities
			np.array([27.5, 12.9, 0.,	# Position x, y, z
			1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
			0., 0., 0.,			# Linear Velocities
			0., 0., 0.]),		# Angular Velocities
			#np.array([5.5, 11.5, 0.,	# Position x, y, z
			#1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
			#0., 0., 0.,			# Linear Velocities
			#0., 0., 0.]),		# Angular Velocities
]

# Test Hallway bend
'''
map_state = [np.array([8.5, 11.2, 0.,	# Position x, y, z
								1., 0., 0., 0.,	# Quaternion w, z, (x, y?)
								0., 0., 0.,			# Linear Velocities
								0., 0., 0.]),		# Angular Velocities
						]
'''
common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': len(map_state),
    'use_mpc': False,
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
    'type': AgentMPEPCTurtlebot,
    'dt': 0.05,
    'conditions': common['conditions'],
    'T': 150,
    'x0': x0s,
    'use_mpc': common['use_mpc'],
    'M': 10,
    'reset_conditions': reset_conditions,
	'goal_conditions': goal_conditions,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [MOBILE_POSITION, MOBILE_ORIENTATION, \
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR],
    'obs_include': [MOBILE_RANGE_SENSOR, MOBILE_ORIENTATION, \
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR],
}

algorithm = {
    'type': AlgorithmPIGPS,
    'conditions': common['conditions'],
    'policy_sample_mode': 'replace',
    'sample_on_policy': True,
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 1.0,
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
    'wu': np.ones(SENSOR_DIMS[ACTION])*5e-5
}

state_cost = {
    'type': CostState,
    'data_types' : {
        MOBILE_ORIENTATION: {
            'wp': np.ones(SENSOR_DIMS[MOBILE_ORIENTATION])*100.,
            'target_state': np.array([0., 0., 0., 1.]),
        },
        MOBILE_VELOCITIES_LINEAR: {
            'wp': np.ones(SENSOR_DIMS[MOBILE_VELOCITIES_LINEAR])*10.,
            'target_state': np.array([1.0, 0., 0.]),
        },
        MOBILE_VELOCITIES_ANGULAR: {
            'wp': np.ones(SENSOR_DIMS[MOBILE_VELOCITIES_ANGULAR])*2.5,
            'target_state': np.array([0., 0., 0.]),
        },
    },
}
#
# obstacle_cost = {
#     'type': CostObstacle,
#     'obstacle_type' : POSITION_NEAREST_OBSTACLE,
#     'position_type': MOBILE_POSITION,
# 		'wp': np.ones(SENSOR_DIMS[MOBILE_POSITION]),
# 		'd_safe': 1.0
# }
#
# algorithm['cost'] = {
#     'type': CostSum,
#     'costs': [action_cost, state_cost, obstacle_cost],
#     'weights': [1.0, 1.0, 50.0],
# }

mpepc_cost = {
    'type': CostMPEPC,
    'obstacle_type' : POSITION_NEAREST_OBSTACLE,
    'position_type': MOBILE_POSITION,
	'potential_type' : POTENTIAL_SCORE,
	'wp': np.ones(SENSOR_DIMS[MOBILE_POSITION]),
	'wp_col' : 10.0,
	'wp_nf' : 1.0,
	'sigma_square': 0.4,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost, mpepc_cost],
    'weights': [1.0, 1.0, 1.0],
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

# algorithm['traj_opt'] = {
#     'type': TrajOptLQRPython,
# }

algorithm['traj_opt'] = {
    'type': TrajOptPI2,
    'kl_threshold': 2.0,
    'covariance_damping': 2.0,
    'min_temperature': 0.001,
}

algorithm['policy_opt'] = {
    'type': PolicyOptCaffe,
    'weights_file_prefix': EXP_DIR + 'policy',
	'iterations': 10000,
    'network_arch_params': {
        'n_layers': 2,
        'dim_hidden': [20],
    },
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': 30,
	'num_samples': 20,
    'common': common,
    'verbose_trials': 0,
    'verbose_policy_trials': 1,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
