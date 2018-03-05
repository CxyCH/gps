/*
 * controllaw_node.cpp
 *
 *  Created on: Feb 19, 2018
 *      Author: thobotics
 */
#include <tf/transform_listener.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include "gps_agent_pkg/EgoGoal.h"
#include "gps_agent_pkg/mpepccontrollaw.h"

using namespace mpepc_local_planner;

tf::TransformListener* tf_;
ros::Publisher cmd_vel_pub_, inter_goal_pub_;
EgoPolar inter_goal_coords_;
double inter_goal_vMax_, inter_goal_k1_, inter_goal_k2_;
double GOAL_DIST_UPDATE_THRESH;
double GOAL_ANGLE_UPDATE_THRESH;
geometry_msgs::Pose local_goal_pose_;
bool is_goal_received = false;
boost::mutex inter_goal_mutex_;
ControlLawSettings settings_;
ControlLaw* cl;

void egoGoalCallback(const gps_agent_pkg::EgoGoal::ConstPtr& ego_goal)
{
	boost::mutex::scoped_lock lock(inter_goal_mutex_);
//    ROS_INFO("EgoGoal: [%f, %f, %f]", ego_goal->r, ego_goal->delta, ego_goal->theta);
    inter_goal_coords_.r = ego_goal->r;
    inter_goal_coords_.delta = ego_goal->delta;
    inter_goal_coords_.theta = ego_goal->theta;
    inter_goal_vMax_ = ego_goal->vMax;
    inter_goal_k1_ = ego_goal->k1;
    inter_goal_k2_ = ego_goal->k2;
}

void globalGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& global_goal_pose)
{
	geometry_msgs::PoseStamped local_pose_stamp;
	try{
		tf_->waitForTransform("/odom", "/map", global_goal_pose->header.stamp, ros::Duration(10.0));
		tf_->transformPose("/odom", *global_goal_pose, local_pose_stamp);
	}catch (tf::TransformException & ex){
		ROS_ERROR("Transform exception 222 : %s", ex.what());
	}

	local_goal_pose_ = local_pose_stamp.pose;

	ROS_INFO("Local goal update: [%f, %f] ",
				local_goal_pose_.position.x, local_goal_pose_.position.y);
	is_goal_received = true;
}

void getCurrentRobotPose(geometry_msgs::Pose &pose){
	tf::StampedTransform tfTransform;
	try {
		tf_->lookupTransform("odom", "base_footprint",
										   ros::Time(0), tfTransform);
	} catch (tf::TransformException& e) {
		ROS_WARN_STREAM_THROTTLE(
		  5.0,
		  "TF lookup from base_footprint to odom failed. Reason: " << e.what());
		return;
	}

	tf::pointTFToMsg(tfTransform.getOrigin(), pose.position);
	tf::quaternionTFToMsg(tfTransform.getRotation(), pose.orientation);
}

void computeVelocity(geometry_msgs::Pose &current_pose, geometry_msgs::Twist &cmd_vel){
	EgoPolar global_goal_coords;
	global_goal_coords = cl->convert_to_egopolar(current_pose, local_goal_pose_);

	ROS_DEBUG("Distance to goal: %f", global_goal_coords.r);
	if (global_goal_coords.r <= GOAL_DIST_UPDATE_THRESH){
		double angle_error = tf::getYaw(current_pose.orientation) - tf::getYaw(local_goal_pose_.orientation);
		angle_error = cl->wrap_pos_neg_pi(angle_error);
		ROS_DEBUG("Angle error: %f", angle_error);

		if (fabs(angle_error) > GOAL_ANGLE_UPDATE_THRESH)
		{
		  cmd_vel.linear.x = 0;
		  if (angle_error > 0)
			cmd_vel.angular.z = -4 * settings_.m_W_TURN;
		  else
			cmd_vel.angular.z = 4 * settings_.m_W_TURN;
		}
		else
		{
//		  ROS_INFO("[MPEPC] Completed normal trajectory following");
//		  goal_reached_ = true;
		  cmd_vel.linear.x = 0;
		  cmd_vel.angular.z = 0;
		}
	} else {
		{
			boost::mutex::scoped_lock lock(inter_goal_mutex_);
			if(inter_goal_coords_.r > 0){ // Used to != -1
				geometry_msgs::Pose inter_goal_pose = cl->convert_from_egopolar(current_pose, inter_goal_coords_);
				cmd_vel = cl->get_velocity_command(current_pose, inter_goal_pose, inter_goal_k1_, inter_goal_k2_, inter_goal_vMax_);

				geometry_msgs::PoseStamped int_goal_stamp;
				int_goal_stamp.header.frame_id = "odom";
				int_goal_stamp.header.stamp = ros::Time(0);
				int_goal_stamp.pose = inter_goal_pose;
				inter_goal_pub_.publish(int_goal_stamp);
			}else{
				cmd_vel.linear.x = 0;
				cmd_vel.angular.z = 0;
			}
		}
	}
}

int main(int argc, char** argv){
    ros::init(argc, argv, "controllaw_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    ros::Rate rate(20);
    tf_ = new tf::TransformListener(ros::Duration(10));

    cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/navi", 1000);
    inter_goal_pub_ = nh.advertise<geometry_msgs::PoseStamped>("inter_goal", 1000);
    ros::Subscriber ego_goal_sub = nh.subscribe("ego_goal", 1, egoGoalCallback);
    ros::Subscriber global_goal_sub = nh.subscribe("global_goal", 1, globalGoalCallback);

    private_nh.param<double>("K_1", settings_.m_K1, 2.0);
	private_nh.param<double>("K_2", settings_.m_K2, 3.0);
	private_nh.param<double>("BETA", settings_.m_BETA, 0.4);
	private_nh.param<double>("LAMBDA", settings_.m_LAMBDA, 2.0);
	private_nh.param<double>("R_THRESH", settings_.m_R_THRESH, 0.05);
	private_nh.param<double>("V_MAX", settings_.m_V_MAX, 1.0);
	private_nh.param<double>("V_MIN", settings_.m_V_MIN, 0.0);
	private_nh.param<double>("W_TURN", settings_.m_W_TURN, 0.2);
	private_nh.param<double>("goal_dist_tol", GOAL_DIST_UPDATE_THRESH, 0.15);
	private_nh.param<double>("goal_angle_tol", GOAL_ANGLE_UPDATE_THRESH, 0.1);

	cl = new ControlLaw(&settings_);
	geometry_msgs::Twist cmd_vel;
	geometry_msgs::Pose current_pose;

	// Control Loop and code
	while (ros::ok())
	{
		getCurrentRobotPose(current_pose);
		if(is_goal_received){ // Make sure received first ever goal
			computeVelocity(current_pose, cmd_vel);
			cmd_vel_pub_.publish(cmd_vel);
		}

		ros::spinOnce();
		rate.sleep();
	}

    return 0;
}
