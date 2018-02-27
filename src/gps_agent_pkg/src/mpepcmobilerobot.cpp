/*
 * mpepcmobilerobot.cpp
 *
 *  Created on: Feb 20, 2018
 *      Author: thobotics
 */
#include "gps_agent_pkg/mpepcmobilerobot.h"

using namespace gps_control;

// Plugin constructor.
MpepcMobileRobot::MpepcMobileRobot()
:MobileRobot()
{
    // Nothing to do here, since all variables are initialized in initialize(...)
	// Some basic variable initialization.
}

// Destructor.
MpepcMobileRobot::~MpepcMobileRobot()
{
    // Nothing to do here, since all instance variables are destructed automatically.
}

void MpepcMobileRobot::init(ros::NodeHandle& n)
{
	// TODO: Set topic name by parameters
	ego_goal_pub_ = n.advertise<gps_agent_pkg::EgoGoal>("/ego_goal", 1);


	// EgoGoal r, delta, theta, vMax
	active_arm_torques_.resize(4);

	// Initialize ROS subscribers/publishers, sensors, and navigation controllers.
	initialize(n);
}

void MpepcMobileRobot::update()
{
	/*
	 * This is the same as the original Mobile Robot */
	// Get current time.
	last_update_time_ = ros::Time::now();

	// Check if this is a controller step based on the current controller frequency.
	controller_counter_++;
	if (controller_counter_ >= controller_step_length_) controller_counter_ = 0;
	bool is_controller_step = (controller_counter_ == 0);

	// Update the sensors and fill in the current step sample.
	update_sensors(last_update_time_,is_controller_step);

    // Update the controllers.
    update_controllers(last_update_time_,is_controller_step);

    /*
     * Publish ego goal instead of raw v, w */
    // Setup action and send to robot
    gps_agent_pkg::EgoGoal ego_goal;
    ego_goal.r = active_arm_torques_[0];
    ego_goal.delta = active_arm_torques_[1];
    ego_goal.theta = active_arm_torques_[2];
    ego_goal.vMax = active_arm_torques_[3];
    ego_goal_pub_.publish(ego_goal);
}


