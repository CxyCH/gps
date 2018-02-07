'''
Created on February 2, 2018

@author: thobotics
'''
import copy
import time
import tf
import numpy as np
import rospy
import math
from threading import Lock
from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_PEDSIM
from gps.sample.sample import Sample
from gps.agent.ros.control_law import ControlLaw

from geometry_msgs.msg import Twist, Pose, Vector3
from nav_msgs.msg import Odometry
from pedsim_msgs.msg import TrackedPersons
from gps.proto.gps_pb2 import MOBILE_POSITION, MOBILE_VELOCITIES_LINEAR, \
                    MOBILE_VELOCITIES_ANGULAR, PEDSIM_AGENT, ACTION


class AgentPedsim(Agent):
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
        config = copy.deepcopy(AGENT_PEDSIM)
        config.update(hyperparams)
        Agent.__init__(self, config)
        if init_node:
            rospy.init_node('gps_agent_pedsim_node')

        self._lock = Lock()
        self.destination = self._hyperparams['sim_goal_state']
        self.pedestrians = []
        self.robot_position = Odometry()
        self.x0 = self._hyperparams['x0']

        self._init_pubs_and_subs()
        self._seq_id = 0  # Used for setting seq in ROS commands.

        r = rospy.Rate(1)
        r.sleep()

    def _init_pubs_and_subs(self):
        self.pub_vel = rospy.Publisher(self._hyperparams['vel_command_topic'], Twist, queue_size=10)
        self.pub_reset = rospy.Publisher(self._hyperparams['reset_command_topic'], Pose, queue_size=10)
        rospy.Subscriber(self._hyperparams['pedsim_agents_topic'], TrackedPersons, self._pedestrian_listener)
        rospy.Subscriber(self._hyperparams['pedbot_position_topic'], Odometry, self._odom_listener)

    def _pedestrian_listener(self, msg):
        self._lock.acquire()
        self.pedestrians = []

        r = max(self._hyperparams['local_width'], self._hyperparams['local_height'])
        robot_position = [self.robot_position.pose.pose.position.x,
                        self.robot_position.pose.pose.position.y]

        pedestrians = msg.tracks

        # Only get Local pedestrian
        for i in range(len(pedestrians)):
            ped_position = [pedestrians[i].pose.pose.position.x,
                            pedestrians[i].pose.pose.position.y]

            dist = math.hypot(robot_position[0] - ped_position[0],
                            robot_position[1] - ped_position[1])

            if (dist <= r):
                self.pedestrians.append(pedestrians[i])

        self._lock.release()

    def _odom_listener(self, msg):
        self._lock.acquire()
        self.robot_position = msg
        self._lock.release()

    def _get_observation(self, condition):
        # Maximum agent
        max_agents = self._hyperparams['max_agents']
        agents = np.ones((max_agents,3)) * 100000

        # Process robot position
        self._lock.acquire()
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
            [self.robot_position.pose.pose.orientation.x,
            self.robot_position.pose.pose.orientation.y,
            self.robot_position.pose.pose.orientation.z,
            self.robot_position.pose.pose.orientation.w])

        robot_position = np.array([self.robot_position.pose.pose.position.x,
            self.robot_position.pose.pose.position.y,
            yaw])

        robot_state = robot_position
        # robot_state = ControlLaw.convert_to_egopolar(robot_position, self.destination[condition])
        robot_linear = np.array([self.robot_position.twist.twist.linear.x,
                                self.robot_position.twist.twist.linear.y])
        robot_angular = np.array([self.robot_position.twist.twist.angular.z])

        # Process pedestrians & compute the observations
        for i in range(len(self.pedestrians)):
            if i > max_agents - 1: break

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [self.pedestrians[i].pose.pose.orientation.x,
                self.pedestrians[i].pose.pose.orientation.y,
                self.pedestrians[i].pose.pose.orientation.z,
                self.pedestrians[i].pose.pose.orientation.w])

            pedestrian = np.array([self.pedestrians[i].pose.pose.position.x,
                            self.pedestrians[i].pose.pose.position.y,
                            yaw])

            agents[i] = ControlLaw.convert_to_egopolar(robot_position, pedestrian)
        self._lock.release()

        # This retrieves the state of the pedsim
        print "State ", robot_state
        state = {MOBILE_POSITION: robot_state,
                 MOBILE_VELOCITIES_LINEAR: robot_linear,
                 MOBILE_VELOCITIES_ANGULAR: robot_angular,
                #  PEDSIM_AGENT: agents}
                 PEDSIM_AGENT: agents.reshape((max_agents*3,))}
        return state

    def _get_next_seq_id(self):
        self._seq_id = (self._seq_id + 1) % (2 ** 32)
        return self._seq_id

    def reset(self, condition):
        """
        Reset the agent for a particular experiment condition.
        Args:
            condition: An index into hyperparams['reset_conditions'].
        """
        # TODO: Change queue instead of changing the position
        #       (do not use deterministic reset)

        # Deterministic reset for robot (not environment)
        x0 = self._hyperparams['sim_x0_state'][condition]
        quaternion = tf.transformations.quaternion_from_euler(0, 0, x0[2])

        pose = Pose()
        pose.position.x = x0[0]
        pose.position.y = x0[1]
        pose.orientation.w = quaternion[3]

        self.pub_reset.publish(pose)

    def sample(self, policy, condition, reset=True, verbose=True, save=True, noisy=True):
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

        # Init sample
        # TODO: Change it to pedsim initial state
        new_sample = Sample(self)
        U = np.zeros([self.T, self.dU])

        # Generate noise.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        # TODO: Experiment on the noise
        # noise = noise*0.01

        # Execute trial.
        frequency = self._hyperparams['frequency']
        rate = rospy.Rate(frequency)
        policy_rate = frequency / 5
        controller_counter = 0
        t = -1
        twist = Twist()

        while not rospy.is_shutdown():
            # Check if this is a controller step based on the current controller frequency.
            if controller_counter >= policy_rate:
                controller_counter = 0

            if controller_counter == 0:
                t += 1
                if t >= self.T:
                    break
                # Update sensor
                # Note: Do not need initialization
                pedsim_X = self._get_observation(condition)
                self._set_sample(new_sample, pedsim_X, t-1)

                # Get controller
                X_t = new_sample.get_X(t=t)
                obs_t = new_sample.get_obs(t=t)
                U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])

            twist.linear.x = U[t, 0]
            twist.angular.z = U[t, 1]

            self.pub_vel.publish(twist)

            controller_counter += 1
            rate.sleep()

        # Stop
        # Move it to reset may be
        twist.linear.x = 0.
        twist.angular.z = 0.
        self.pub_vel.publish(twist)

        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _set_sample(self, sample, pedsim_X, t):
        for sensor in pedsim_X.keys():
            sample.set(sensor, np.array(pedsim_X[sensor]), t=t+1)
