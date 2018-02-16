""" Default configuration and hyperparameters for agent objects. """
import logging

import numpy as np


LOGGER = logging.getLogger(__name__)


# Agent
AGENT = {
    'dH': 0,
    'x0var': 0,
    'noisy_body_idx': np.array([]),
    'noisy_body_var': np.array([]),
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'smooth_noise': True,
    'smooth_noise_var': 2.0,
    'smooth_noise_renormalize': True,
    'use_mpc': False
}


try:
    import rospkg

    import roslib

    roslib.load_manifest('gps_agent_pkg')

    # AgentROS
    AGENT_ROS = {
        #TODO: It might be worth putting this in JSON/yaml format so C++
        #      can read it.
        'trial_command_topic': 'gps_controller_trial_command',
        'reset_command_topic': 'gps_controller_position_command',
        'relax_command_topic': 'gps_controller_relax_command',
        'data_request_topic': 'gps_controller_data_request',
        'sample_result_topic': 'gps_controller_report',
        'trial_timeout': 20,  # Give this many seconds for a trial.
        'reset_conditions': [],  # Defines reset modes + positions for
                                 # trial and auxiliary arms.
        'frequency': 20,
        'end_effector_points': np.array([]),
        #TODO: Actually pass in low gains and high gains and use both
        #      for the position controller.
        'pid_params': np.array([
            2400.0, 0.0, 18.0, 4.0,
            1200.0, 0.0, 20.0, 4.0,
            1000.0, 0.0, 6.0, 4.0,
            700.0, 0.0, 4.0, 4.0,
            300.0, 0.0, 6.0, 2.0,
            300.0, 0.0, 4.0, 2.0,
            300.0, 0.0, 2.0, 2.0
        ]),
    }

    AGENT_TURTLEBOT = {
        #TODO: Check is needed to change these name ?
        'trial_command_topic': 'gps_controller_trial_command',
        'reset_command_topic': 'gps_controller_navigation_command',
        'data_request_topic': 'gps_controller_data_request',
        'sample_result_topic': 'gps_controller_report',
        'trial_timeout': 20,  # Give this many seconds for a trial.
        'reset_conditions': [],  # Defines reset modes + positions for
                                 # trial and auxiliary arms.
        'frequency': 20,
    }

    AGENT_PEDSIM = {
        'vel_command_topic': '/cmd_vel',
        'reset_command_topic': '/cmd_pose',
        'robot_mode_topic': '/robot_mode',
        'pedbot_position_topic': '/pedsim/robot_position',
        'expert_position_topic': '/pedsim/expert_position',
        'pedsim_agents_topic': '/pedsim/tracked_persons',
        'max_agents': 10,  # Give this many seconds for a trial.
        'sim_x0_state': [],
        'sim_goal_state': [],
        'local_width': 3.0,
        'local_height': 3.0,
        'frequency': 20,
    }
except ImportError as e:
    AGENT_ROS = {}
    LOGGER.debug('No ROS enabled: %s', e)
except rospkg.common.ResourceNotFound as e:
    AGENT_ROS = {}
    LOGGER.debug('No gps_agent_pkg: %s', e)


# AgentMuJoCo
AGENT_MUJOCO = {
    'substeps': 1,
    'camera_pos': np.array([2., 3., 2., 0., 0., 0.]),
    'image_width': 640,
    'image_height': 480,
    'image_channels': 3,
    'meta_include': []
}

AGENT_BOX2D = {
    'render': True,
    'world_info': None
}
