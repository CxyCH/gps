'''
Created on Jul 4, 2017

@author: thobotics
'''
import copy
import time
import numpy as np

import rospy

from gps.agent.agent import Agent
from gps.sample.sample import Sample
from gps.agent.config import AGENT_PEDBOT
from gps.agent.agent_utils import generate_noise, setup
from gps.proto.gps_pb2 import MOBILE_POSITION, MOBILE_ORIENTATION, \
                MOBILE_VELOCITIES_LINEAR, MOBILE_VELOCITIES_ANGULAR, ACTION, \
                POSITION_NEAREST_OBSTACLE
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from tf import TransformListener


class AgentPedbot(Agent):
    '''
    classdocs
    '''


    def __init__(self, hyperparams, init_node=True):
        '''
        Constructor
        '''
        """
        Initialize agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
        """
        config = copy.deepcopy(AGENT_PEDBOT)
        config.update(hyperparams)
        Agent.__init__(self, config)
        if init_node:
            rospy.init_node('gps_agent_pedbot_node')
        self._init_pubs_and_subs()
        self._seq_id = 0  # Used for setting seq in ROS commands.

        self.x0 = self._hyperparams['x0']
        # TODO: Init with x0 in many condidtion
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.
        self.move_cmd.linear.y = 0.
        self.move_cmd.angular.z = 0.
        
        self.rate = rospy.Rate(self._hyperparams['frequency'])
        self.tf = TransformListener()
        
        r = rospy.Rate(1)
        r.sleep()
        
    def odometryCb(self,msg):
        self.msg = msg
        #print msg.pose.pose
    
    def run_next(self, action):
        self.move_cmd = Twist() 
        self.move_cmd.linear.x = action[0]
        self.move_cmd.linear.y = action[1]
        self.move_cmd.angular.z = action[2]
        self._cmd_vel.publish(self.move_cmd)
    
    def get_state(self):
        '''
        odom_position = np.array([self.msg.pose.pose.position.x, self.msg.pose.pose.position.y,
                                       self.msg.pose.pose.position.z])
        # TODO Check that
        odom_orientation = np.array([self.msg.pose.pose.orientation.x, self.msg.pose.pose.orientation.y,
                                       self.msg.pose.pose.orientation.z, self.msg.pose.pose.orientation.w])
        odom_vel_linear = np.array([self.msg.twist.twist.linear.x, self.msg.twist.twist.linear.y,
                                       self.msg.twist.twist.linear.z]) 
        odom_vel_angular = np.array([self.msg.twist.twist.angular.x, self.msg.twist.twist.angular.y,
                                       self.msg.twist.twist.angular.z])
        '''
        if self.tf.frameExists("/odom") and self.tf.frameExists("/base_footprint"):
            t = self.tf.getLatestCommonTime("/odom", "/base_footprint")
            position, quaternion = self.tf.lookupTransform("/odom", "/base_footprint", t)
            
            odom_position = np.array(position)
            # TODO Check that
            odom_orientation = np.array(quaternion)
            odom_vel_linear = np.array([self.move_cmd.linear.x, self.move_cmd.linear.y,
                                       self.move_cmd.linear.z]) 
            odom_vel_angular = np.array([self.move_cmd.angular.x, self.move_cmd.angular.y,
                                       self.move_cmd.angular.z])
            #print odom_position, odom_orientation, odom_vel_linear, odom_vel_angular
            
        state = {MOBILE_POSITION: odom_position,
                 MOBILE_ORIENTATION: odom_orientation,
                 MOBILE_VELOCITIES_LINEAR: odom_vel_linear,
                 MOBILE_VELOCITIES_ANGULAR: odom_vel_angular}

        return state
    
    def _init_pubs_and_subs(self):
        self._cmd_vel = rospy.Publisher("/pedbot/control/cmd_vel", Twist)
        self._cmd_pose = rospy.Publisher("/pedbot/control/cmd_pose", Pose)
        self._odom = rospy.Subscriber("/pedbot/robot_position", Odometry, self.odometryCb)
        
    def _get_next_seq_id(self):
        self._seq_id = (self._seq_id + 1) % (2 ** 32)
        return self._seq_id
    
    def reset(self, condition):
        condition_data = self._hyperparams['reset_conditions'][condition]
        reset_command = Pose()
        reset_command.position.x = condition_data[0]
        reset_command.position.y = condition_data[1]
        reset_command.position.z = condition_data[2]
        reset_command.orientation.x = condition_data[3]
        reset_command.orientation.y = condition_data[4]
        reset_command.orientation.z = condition_data[5]
        reset_command.orientation.w = condition_data[6]
        self._cmd_pose.publish(reset_command)
        time.sleep(2.0)  # useful for the real robot, so it stops completely
    
    def sample(self, policy, condition, reset=True, verbose=True, save=True, noisy=True, record_image=False, record_gif=None, record_gif_fps=None):
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
        if reset:
            self.reset(condition)
        
        b2d_X = self.get_state()
        new_sample = self._init_sample(b2d_X)
        U = np.zeros([self.T, self.dU])
        
        #raw_input("Hello")
        # Generate noise.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        noise = noise*0.01
        
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
            if (t+1) < self.T:
                self.run_next(U[t, :])
                self.rate.sleep()
                b2d_X = self.get_state()
                self._set_sample(new_sample, b2d_X, t)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample
    
    def _init_sample(self, b2d_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, b2d_X, -1)
        return sample

    def _set_sample(self, sample, b2d_X, t):
        for sensor in b2d_X.keys():
            sample.set(sensor, np.array(b2d_X[sensor]), t=t+1)