""" Hyperparameters for PR2 trajectory optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.agent.box2d.point_mass_world_obstacle_mobile import PointMassWorldObstacleMobile
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
from copy import copy

SENSOR_DIMS = {
    MOBILE_POSITION: 3,
    MOBILE_ORIENTATION: 4,
    MOBILE_VELOCITIES_LINEAR: 3,
    MOBILE_VELOCITIES_ANGULAR: 3,
    ACTION: 3
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/box2d_pointmass_mobile/'

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
    'use_mpc': False,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

world_info = {
		'obstacles': [np.array([-5, 13, 4, 1]),
			 np.array([10, 13, 4, 1]),
			 np.array([-9, 20, 4, 1]),
			 np.array([15, 20, 4, 1]),
			 np.array([3, 20, 1.5, 1]),
			],
}
world_info = {'obstacles':[np.array([-1, 15, 4, 1])]}

agent = {
    'type': AgentBox2D,
    'target_state' : np.array([3, 35, 0]),
    "world" : PointMassWorldObstacleMobile,
    'world_info': world_info,
    #'render' : False,
    'x0': [np.array([0, 5, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0])],
    'rk': 0,
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 100,
    'use_mpc': common['use_mpc'],
    'M': 5,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [MOBILE_POSITION, MOBILE_ORIENTATION, \
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR],
    'obs_include': [MOBILE_POSITION, MOBILE_ORIENTATION, \
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR],
}


algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'use_mpc': common['use_mpc'],
    'iterations': 20,
		'target_end_effector': [agent["target_state"]],
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
            'target_state': np.array([0., 0., 1., 0.]),
        },
        MOBILE_VELOCITIES_LINEAR: {
            'wp': np.ones(SENSOR_DIMS[MOBILE_VELOCITIES_LINEAR])*100.0,
            'target_state': np.array([0., 5., 0.]),
        },
        MOBILE_VELOCITIES_ANGULAR: {
            'wp': np.ones(SENSOR_DIMS[MOBILE_VELOCITIES_ANGULAR])*2.5,
            'target_state': np.array([0., 0., 0.]),
        },
    },
}
'''
algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1.0, 1.0],
}
'''

obstacle_cost = {
    'type': CostObstacle,
    'obstacle_type' : POSITION_NEAREST_OBSTACLE,
    'position_type': MOBILE_POSITION,
		'wp': np.ones(SENSOR_DIMS[MOBILE_POSITION]),
		'd_safe': 0.5
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost, obstacle_cost],
    'weights': [1.0, 1.0, 25.0],
}
#'''

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
    'num_samples': 20,
    'dQ': algorithm['init_traj_distr']['dQ'],
}

common['info'] = generate_experiment_info(config)
