'''
Created on Mar 3, 2017

@author: thobotics
'''
import copy
import time
import numpy as np

import rospy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_TURTLEBOT
from gps.agent.ros.ros_utils import ServiceEmulator, msg_to_sample, \
        policy_to_msg, tf_policy_to_action_msg, tf_obs_msg_to_numpy
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM
from gps_agent_pkg.msg import TrialCommand, SampleResult, NavigationCommand, \
        DataRequest
from __builtin__ import raw_input


class AgentTurtlebot(Agent):
    """
    All communication between the algorithms and ROS is done through
    this class.
    """
    def __init__(self, hyperparams, init_node=True):
        """
        Initialize agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
        """
        config = copy.deepcopy(AGENT_TURTLEBOT)
        config.update(hyperparams)
        Agent.__init__(self, config)
        if init_node:
            rospy.init_node('gps_agent_turtlebot_node')
        self._init_pubs_and_subs()
        self._seq_id = 0  # Used for setting seq in ROS commands.

        conditions = self._hyperparams['conditions']

        self.x0 = []
        '''
        TOOD: CHECK THIS
        for field in ('x0', 'ee_points_tgt', 'reset_conditions'):
            self._hyperparams[field] = setup(self._hyperparams[field],
                                             conditions)
        '''
        self.x0 = self._hyperparams['x0']

        r = rospy.Rate(1)
        r.sleep()

        #self.use_tf = False
        #self.observations_stale = True
        
    def _init_pubs_and_subs(self):
        self._trial_service = ServiceEmulator(
            self._hyperparams['trial_command_topic'], TrialCommand,
            self._hyperparams['sample_result_topic'], SampleResult
        )
        self._reset_service = ServiceEmulator(
            self._hyperparams['reset_command_topic'], NavigationCommand,
            self._hyperparams['sample_result_topic'], SampleResult
        )
        self._data_service = ServiceEmulator(
            self._hyperparams['data_request_topic'], DataRequest,
            self._hyperparams['sample_result_topic'], SampleResult
        )
        
    def _get_next_seq_id(self):
        self._seq_id = (self._seq_id + 1) % (2 ** 32)
        return self._seq_id
    
    def get_data(self):
        """
        Request for the most recent value for data/sensor readings.
        Returns entire sample report (all available data) in sample.
        """
        request = DataRequest()
        request.id = self._get_next_seq_id()
        request.arm = TRIAL_ARM
        request.stamp = rospy.get_rostime()
        result_msg = self._data_service.publish_and_wait(request)
        # TODO - Make IDs match, assert that they match elsewhere here.
        sample = msg_to_sample(result_msg, self)
        return sample

    # TODO: CHECK THIS
    def reset(self, condition):
        """
        Reset the agent for a particular experiment condition.
        Args:
            condition: An index into hyperparams['reset_conditions'].
        """
        condition_data = self._hyperparams['reset_conditions'][condition]
        reset_command = NavigationCommand()
        reset_command.position = condition_data[:2]
        reset_command.quaternion = condition_data[3:7]
        timeout = self._hyperparams['trial_timeout']
        reset_command.id = self._get_next_seq_id()
        self._reset_service.publish_and_wait(reset_command, timeout=timeout)
        time.sleep(2.0)  # useful for the real robot, so it stops completely

    # TODO: CHECK THIS
    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Reset and execute a policy and collect a sample.
        Args:
            policy: A Policy object.
            condition: Which condition setup to run.
            verbose: Unused for this agent.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        Returns:
            sample: A Sample object.
        """
        
        self.reset(condition)
        # Generate noise.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Execute trial.
        trial_command = TrialCommand()
        trial_command.id = self._get_next_seq_id()
        trial_command.controller = policy_to_msg(policy, noise)
        trial_command.T = self.T
        trial_command.id = self._get_next_seq_id()
        trial_command.frequency = self._hyperparams['frequency']
        # ee_points and ee_points_tgt is uneccesary for mobile robot
        trial_command.ee_points = []
        trial_command.ee_points_tgt = []
        trial_command.state_datatypes = self._hyperparams['state_include']
        trial_command.obs_datatypes = self._hyperparams['state_include']
        
        sample_msg = self._trial_service.publish_and_wait(
            trial_command, timeout=self._hyperparams['trial_timeout']
        )

        sample = msg_to_sample(sample_msg, self)
        if save:
            self._samples[condition].append(sample)
        return sample