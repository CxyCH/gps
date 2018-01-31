""" Hyperparameters for MJC 2D navigation policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.agent.box2d.point_mass_world_obstacle_mobile import PointMassWorldObstacleMobile
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_ioc_tf import CostIOCTF
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_obstacles import CostObstacle
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.proto.gps_pb2 import MOBILE_POSITION, MOBILE_ORIENTATION, \
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR, ACTION, \
				POSITION_NEAREST_OBSTACLE
from gps.gui.config import generate_experiment_info


SENSOR_DIMS = {
    MOBILE_POSITION: 3,
    MOBILE_ORIENTATION: 4,
    MOBILE_VELOCITIES_LINEAR: 3,
    MOBILE_VELOCITIES_ANGULAR: 3,
    POSITION_NEAREST_OBSTACLE: 3,
    ACTION: 3
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = os.path.dirname(__file__)+'/'
DEMO_DIR = BASE_DIR + '/../experiments/box2d_pointmass_mobile/'
SUPERVISED_DIR = BASE_DIR + '/../experiments/mjc_pointmass_wall_supervised/'
DEMO_CONDITIONS = 1

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'demo_exp_dir': DEMO_DIR,
    'demo_controller_file': DEMO_DIR + 'data_files/algorithm_itr_19.pkl',
    'supervised_exp_dir': SUPERVISED_DIR,
    'NN_demo_file': EXP_DIR + 'data_files/demo_NN.pkl',
    'LG_demo_file': EXP_DIR + 'data_files/demo_LG.pkl',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
    'demo_conditions': 3,
    'nn_demo': False,
    'use_mpc': False,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentBox2D,
    'target_state' : np.array([3, 35, 0]),
    "world" : PointMassWorldObstacleMobile,
    'world_info': {'obstacles':[np.array([-1, 15, 4, 1])]},
    'render' : False,
    'x0': [np.array([0, 5, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0])],
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
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR, POSITION_NEAREST_OBSTACLE],
    'obs_include': [MOBILE_POSITION, MOBILE_ORIENTATION, \
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR, POSITION_NEAREST_OBSTACLE],
}

demo_agent = {
    'type': AgentBox2D,
    'target_state' : np.array([3, 35, 0]),
    "world" : PointMassWorldObstacleMobile,
    'world_info': {'obstacles':[np.array([-1, 15, 4, 1])]},
    #'render' : True,
    'x0': [np.array([0, 5, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0])],
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
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR, POSITION_NEAREST_OBSTACLE],
    'obs_include': [MOBILE_POSITION, MOBILE_ORIENTATION, \
				MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR, POSITION_NEAREST_OBSTACLE],
}

unlabeled_agent = {}

algorithm = {
    'type': AlgorithmTrajOpt,
    'ioc' : 'ICML',
    'demo_distr_empest': True,
    'conditions': common['conditions'],
    'iterations': 80,
    'kl_step': 1.0,
    'min_step_mult': 0.01,
    'max_step_mult': 4.0,
    'max_ent_traj': 1.0,
    'num_demos': 10,
    'target_end_effector': np.array([1.3, 0.5, 0.]),
    
    'demo_var_mult':1.0,
    'target_end_effector': [agent["target_state"]],
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 5.0,
    'pos_gains': 1.0,
    'vel_gains_mult': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}

algorithm['cost'] = {
    'type': CostIOCTF,
    'wu': np.ones(SENSOR_DIMS[ACTION])*1e-3,
    'dO': 16,
    'T': agent['T'],
    'iterations': 2000,
    'demo_batch_size': 5,
    'sample_batch_size': 5,
    'ioc_loss': algorithm['ioc'],
    
    'approximate_lxx': False,
    'dim_hidden': 1,
    #'dim_hidden': 42,
    
    'feature' : {
        'state_types' : {
            MOBILE_ORIENTATION:{
                'idx_start': 3,
                'len': 4,
                'target_state': np.array([0., 0., 1., 0.]),
            },
            MOBILE_VELOCITIES_LINEAR: {
                'idx_start': 7,
                'len': 3,
                'target_state': np.array([0., 5., 0.]),
            },
            MOBILE_VELOCITIES_ANGULAR: {
                'idx_start': 10,
                'len': 3,
                'target_state': np.array([0., 0., 0.]),
            },
        },
        'obs_types' : {
                'obstacle_type' : POSITION_NEAREST_OBSTACLE,
                'd_safe': 0.5
        }
    }
}

action_cost = {
    'type': CostAction,
    'wu': np.ones(SENSOR_DIMS[ACTION])*5e-3
}

state_cost = {
    'type': CostState,
    'data_types' : {
        MOBILE_ORIENTATION: {
            'wp': np.ones(SENSOR_DIMS[MOBILE_ORIENTATION])*1.,
            'target_state': np.array([0., 0., 1., 0.]),
        },
        MOBILE_VELOCITIES_LINEAR: {
            'wp': np.ones(SENSOR_DIMS[MOBILE_VELOCITIES_LINEAR])*1.0,
            'target_state': np.array([0., 5., 0.]),
        },
        MOBILE_VELOCITIES_ANGULAR: {
            'wp': np.ones(SENSOR_DIMS[MOBILE_VELOCITIES_ANGULAR])*0.025,
            'target_state': np.array([0., 0., 0.]),
        },
    },
}

obstacle_cost = {
    'type': CostObstacle,
    'obstacle_type' : POSITION_NEAREST_OBSTACLE,
    'position_type': MOBILE_POSITION,
		'wp': np.ones(SENSOR_DIMS[MOBILE_POSITION]),
		'd_safe': 0.5
}

algorithm['gt_cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost, obstacle_cost],
    'weights': [1.0, 1.0, 0.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 2,
        'min_samples_per_cluster': 20,
        'max_samples': 20,
    }
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': agent['obs_include'],
        'n_layers': 3,
        'dim_hidden': 40,
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': tf_network,
    'fc_only_iterations': 2000,
    'init_iterations': 1000,
    'iterations': 1000,  # was 100
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 10,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'demo_agent': demo_agent,
    'unlabeled_agent': unlabeled_agent,
    'gui_on': True,
    'algorithm': algorithm,
    'arecord_gif': {
        'gif_dir': os.path.join(common['data_files_dir'], 'gifs'),
        'gifs_per_condition': 1,
        'record_every': 5,
        'save_traj_samples': False,
        'fps': 40,
    }
}
seed = 2

common['info'] = generate_experiment_info(config)
