'''
Created on Mar 3, 2017

@author: thobotics
'''
import copy
import time
import numpy as np

import rospy

from gps.agent.ros.agent_turtlebot import AgentTurtlebot
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_MPEPEC_TURTLEBOT
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from __builtin__ import raw_input


class AgentMPEPCTurtlebot(AgentTurtlebot):
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
        config = copy.deepcopy(AGENT_MPEPEC_TURTLEBOT)
        config.update(hyperparams)
        AgentTurtlebot.__init__(self, config)

    def _init_pubs_and_subs(self):
        AgentTurtlebot._init_pubs_and_subs(self)
        self._global_goal = rospy.Publisher("global_goal", PoseStamped)

    def reset(self, condition):
        """
        Reset the agent for a particular experiment condition.
        Args:
            condition: An index into hyperparams['goal_conditions'].
        """
        AgentTurtlebot.reset(self, condition)

        # Publish global plan
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "/map"

        condition_data = self._hyperparams['goal_conditions'][condition]
        pose.pose.position.x = condition_data[0]
        pose.pose.position.y = condition_data[1]
        pose.pose.position.z = condition_data[2]
        pose.pose.orientation.x = condition_data[3]
        pose.pose.orientation.y = condition_data[4]
        pose.pose.orientation.z = condition_data[5]
        pose.pose.orientation.w = condition_data[6]
        self._global_goal.publish(pose)
