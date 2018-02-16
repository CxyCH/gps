""" Hyperparameters for Box2d Point Mass task with PIGPS."""
from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.ros.agent_pedsim import AgentPedsim
from gps.algorithm.algorithm_pigps import AlgorithmPIGPS
from gps.algorithm.algorithm_pigps import AlgorithmMDGPS
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_pedestrian import CostPedestrian
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.algorithm.policy.lin_gauss_init import init_pd, init_demo_lg
from gps.proto.gps_pb2 import MOBILE_POSITION, MOBILE_ORIENTATION, \
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR, ACTION, \
				POSITION_NEAREST_OBSTACLE, PEDSIM_AGENT
from gps.gui.config import generate_experiment_info
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython

SENSOR_DIMS = {
    MOBILE_POSITION: 3,
    MOBILE_VELOCITIES_LINEAR: 2,
    MOBILE_VELOCITIES_ANGULAR: 1,
    PEDSIM_AGENT: 10*3,
    ACTION: 2
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/pedsim_pigps_example/'

common = {
    'experiment_name': 'pedsim_pigps_example' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentPedsim,
    'sim_x0_state' : [np.array([0., 4., 0.])],
    'sim_goal_state' : [np.array([30., 6., 0.])],
    'render' : False,
	'max_agents': 10,
    'x0': [np.append(np.array([
		0., 0., 0.,
		0., 0., 0.
	]), np.zeros(10*3))],
    'rk': 0,
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [MOBILE_POSITION, MOBILE_VELOCITIES_LINEAR, \
                    MOBILE_VELOCITIES_ANGULAR, PEDSIM_AGENT],
    'obs_include': [MOBILE_POSITION, MOBILE_VELOCITIES_LINEAR, \
                    MOBILE_VELOCITIES_ANGULAR, PEDSIM_AGENT],
    # 'smooth_noise_var': 3.0,
}

algorithm = {
    'type': AlgorithmPIGPS,
    'conditions': common['conditions'],
    'policy_sample_mode': 'replace',
    'sample_on_policy': True,
}

# algorithm['init_traj_distr'] = {
#     'type': init_pd,
#     'init_var': 1.0,
#     'pos_gains': 0.0,
#     'dQ': SENSOR_DIMS[ACTION],
#     'dt': agent['dt'],
#     'T': agent['T'],
# }

algorithm['init_traj_distr'] = {
    'type': init_demo_lg,
	'iteration': 5,
	'init_var': 3.0,
	'data_files_dir': common['data_files_dir'],
	'condition': [i for i in range(common['conditions'])],
	'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.ones(SENSOR_DIMS[ACTION])*5e-5
}

state_cost = {
    'type': CostState,
    'data_types' : {
		# MOBILE_POSITION: {
		# 	'wp': np.array([0.0, 0.0, 50.0]),
		# 	'target_state': np.array([35.0, 2.0, 0.0]),
		# },
		MOBILE_POSITION: {
			'wp': np.array([0.05, 10.0, 10.0]),
			'target_state': np.array([0.0, 0.0, 0.0]),
		},
        # MOBILE_VELOCITIES_LINEAR: {
        #     'wp': np.ones(SENSOR_DIMS[MOBILE_VELOCITIES_LINEAR])*1.0,
        #     'target_state': np.array([2.0, 0.]),
        # },
        # MOBILE_VELOCITIES_ANGULAR: {
        #     'wp': np.ones(SENSOR_DIMS[MOBILE_VELOCITIES_ANGULAR])*100.0,
        #     'target_state': np.array([0.]),
        # },
    },
}

pedestrian_cost = {
    'type': CostPedestrian,
    'pedestrian_type' : PEDSIM_AGENT,
    'max_agents': agent['max_agents'],
	'wp': np.array([100.0, 0.0, 0.0]),
	'd_safe': 1.0
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost, pedestrian_cost],
    'weights': [1.0, 1.0, 3.0],
}

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
    'type': PolicyPrior,
}

config = {
    'iterations': 30,
    'num_samples': 20,
    'common': common,
    'verbose_trials': 1,
    'verbose_policy_trials': 0,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
