/*
 * navcontroller.cpp
 *
 *  Created on: Mar 8, 2017
 *      Author: thobotics
 */
#include "gps_agent_pkg/navcontroller.h"
#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/util.h"

using namespace gps_control;

// Constructor.
NavController::NavController(ros::NodeHandle& n)
    : Controller(n, gps::TRIAL_ARM, 0)
{
	report_waiting = false;
	// TODO: Change topic names to parameters or something
	nav_pub_ = n.advertise<geometry_msgs::Pose>("/cmd_pose", 1);
}

// Destructor.
NavController::~NavController()
{
}

// Update the controller (take an action).
void NavController::update(RobotPlugin *plugin, ros::Time current_time, boost::scoped_ptr<Sample>& sample, Eigen::VectorXd &torques)
{
	// TODO: Send target pose to move_base and waiting it finished
	if(finished_){
		torques = Eigen::VectorXd::Zero(torques.rows());
	}
}

// Configure the controller.
void NavController::configure_controller(OptionsMap &options)
{
	// This sets the target position.
	ROS_INFO_STREAM("Received controller configuration");
	// needs to report when finished
	report_waiting = true;
	finished_ = false;

	// Update target pose of robot
	position_ = boost::get<Eigen::VectorXd>(options["position"]);
	orientation_ = boost::get<Eigen::VectorXd>(options["orientation"]);

	// NOTE: This pose for Stage frame (map).
	// Its coordinate (yzx ??, x,y = 0.707)
	// is not the same as odom (zyx, w,z = 0.707)
	geometry_msgs::Pose stage_pose;
	stage_pose.position.x = position_(0);
	stage_pose.position.y = position_(1);
	stage_pose.position.z = position_(2);
	stage_pose.orientation.x = orientation_(0);
	stage_pose.orientation.y = orientation_(1);
	stage_pose.orientation.z = orientation_(2);
	stage_pose.orientation.w = orientation_(3);
	nav_pub_.publish(stage_pose);

	// TODO: It totally wrong, must check position in is_finished
	finished_ = true;
}

// Check if controller is finished with its current task.
bool NavController::is_finished() const
{
	return finished_; // TODO: CHECK Reach goal and return
}

// Reset the controller -- this is typically called when the controller is turned on.
void NavController::reset(ros::Time time)
{
	// Clear update time.
	// last_update_time_ = ros::Time(0.0);
}
