/*
 * mpepcmobilerobot.cpp
 *
 *  Created on: Feb 20, 2018
 *      Author: thobotics
 */
#include "gps_agent_pkg/mpepcmobilerobot.h"
#include "gps_agent_pkg/trialcontroller.h"

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

void MpepcMobileRobot::init(ros::NodeHandle& n, bool use_mpepc)
{
	use_mpepc_ = use_mpepc;

	// TODO: Set topic name by parameters
	ego_goal_pub_ = n.advertise<gps_agent_pkg::EgoGoal>("/ego_goal", 1);


	// EgoGoal r, delta, theta, vMax
	active_arm_torques_.resize(4);

	// Initialize ROS subscribers/publishers, sensors, and navigation controllers.
	initialize(n);

	if(use_mpepc_){
		ros::NodeHandle private_nh("~");
		// ControlLaw parameters
		private_nh.param<double>("K_1", settings_.m_K1, 2.0);
		private_nh.param<double>("K_2", settings_.m_K2, 3.0);
		private_nh.param<double>("BETA", settings_.m_BETA, 0.4);
		private_nh.param<double>("LAMBDA", settings_.m_LAMBDA, 2.0);
		private_nh.param<double>("R_THRESH", settings_.m_R_THRESH, 0.05);
		private_nh.param<double>("V_MAX", settings_.m_V_MAX, 1.0);
		private_nh.param<double>("V_MIN", settings_.m_V_MIN, 0.0);
		private_nh.param<double>("W_TURN", settings_.m_W_TURN, 0.0);

		// Minimization parameters
		private_nh.param<double>("TIME_HORIZON", TIME_HORIZON, 3.0);
		private_nh.param<double>("DELTA_SIM_TIME", DELTA_SIM_TIME, 0.2);
		private_nh.param<double>("SAFETY_ZONE", SAFETY_ZONE, 0.225);
		private_nh.param<double>("C1", C1, 0.05);
		private_nh.param<double>("C2", C2, 1.00);
		private_nh.param<double>("C3", C3, 0.05);
		private_nh.param<double>("C4", C4, 0.05);
		private_nh.param<double>("SIGMA", SIGMA, 0.40);

		l_plan_pub_ = private_nh.advertise<geometry_msgs::PoseArray>("mpepc_local_plan", 1);

		cl = new mpepc_local_planner::ControlLaw(&settings_);
		mpepc_sensor_ = static_cast<MobileRobotSensor*>(sensors_[0].get()); // Only 1 sensor

	}
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
    bool trial_init = trial_controller_ != NULL && trial_controller_->is_configured() && controller_initialized_;
    if (use_mpepc_ && trial_init && !trial_controller_->is_finished()){
    	if(is_controller_step){
			sim_current_pose_ = mpepc_sensor_->getCurrentRobotPose();
			find_intermediate_goal_params(&ego_goal_);

			// Send back to gps trainer
			active_arm_torques_[0] = ego_goal_.r;
			active_arm_torques_[1] = ego_goal_.delta;
			active_arm_torques_[2] = ego_goal_.theta;
			active_arm_torques_[3] = ego_goal_.vMax;

//			ROS_INFO("Step %d - %f %f", trial_controller_->get_step_counter()-1, active_arm_torques_[0], active_arm_torques_[1]);

			// Set the torques for the sample
			current_time_step_sample_->set_data(trial_controller_->get_step_counter()-1,
					gps::ACTION,active_arm_torques_,active_arm_torques_.size(),SampleDataFormatDouble);
    	}
    }else{
		ego_goal_.r = active_arm_torques_[0];
		ego_goal_.delta = active_arm_torques_[1];
		ego_goal_.theta = active_arm_torques_[2];
		ego_goal_.vMax = active_arm_torques_[3];
    }

    if(is_controller_step){
    	l_plan_pub_.publish(get_trajectory_viz(ego_goal_));
    	ego_goal_pub_.publish(ego_goal_);
    }
}

geometry_msgs::PoseArray MpepcMobileRobot::get_trajectory_viz(gps_agent_pkg::EgoGoal new_coords){
	geometry_msgs::PoseArray viz_plan;
	viz_plan.header.stamp = ros::Time::now();
	viz_plan.header.frame_id = "odom";
	viz_plan.poses.resize(1);

	geometry_msgs::Pose sim_pose = sim_current_pose_;

	mpepc_local_planner::EgoPolar sim_goal;
	sim_goal.r = new_coords.r;
	sim_goal.delta = new_coords.delta;
	sim_goal.theta = new_coords.theta;

	geometry_msgs::Pose current_goal = cl->convert_from_egopolar(sim_pose, sim_goal);

	double sim_clock = 0.0;

	geometry_msgs::Twist sim_cmd_vel;
	double current_yaw = tf::getYaw(sim_pose.orientation);

	while (sim_clock < TIME_HORIZON)
	{
	  sim_cmd_vel = cl->get_velocity_command(sim_goal, new_coords.k1, new_coords.k2, new_coords.vMax);

	  // Update pose
	  current_yaw = current_yaw + (sim_cmd_vel.angular.z * DELTA_SIM_TIME);
	  sim_pose.position.x = sim_pose.position.x + (sim_cmd_vel.linear.x * DELTA_SIM_TIME * cos(current_yaw));
	  sim_pose.position.y = sim_pose.position.y + (sim_cmd_vel.linear.x * DELTA_SIM_TIME * sin(current_yaw));
	  sim_pose.orientation = tf::createQuaternionMsgFromYaw(current_yaw);
	  viz_plan.poses.push_back(sim_pose);

	  sim_goal = cl->convert_to_egopolar(sim_pose, current_goal);

	  sim_clock = sim_clock + DELTA_SIM_TIME;
	}

	return viz_plan;
}

double MpepcMobileRobot::sim_trajectory(double r, double delta, double theta, double vMax){
	// Get robot pose from local cost map
	// DONE: Change this to getCurrentRobotPose.
	// 1. Why need to change this? Ans: May be it help for faster reading current pose,
	// but not sure because getCurrentRobotPose use transform instead of callback in Odom.
	// 2. Why not change it now? Ans: Because it may need to change other file: control_law, ...

	double time_horizon = TIME_HORIZON;
	geometry_msgs::Pose sim_pose = sim_current_pose_;

	mpepc_local_planner::EgoPolar sim_goal;
	sim_goal.r = r == 0 ? 0.001 : r;
	sim_goal.delta = delta;
	sim_goal.theta = theta;

	geometry_msgs::Pose current_goal = cl->convert_from_egopolar(sim_pose, sim_goal);

	double SIGMA_DENOM = pow(SIGMA, 2);

	double sim_clock = 0.0;

	geometry_msgs::Twist sim_cmd_vel;
	double current_yaw = tf::getYaw(sim_pose.orientation);
	geometry_msgs::Point collisionPoint;
	bool collision_detected = false;

	double expected_progress = 0.0;
	double expected_action = 0.0;
	double expected_collision = 0.0;

	double nav_fn_t0 = 0;
	double nav_fn_t1 = 0;
	double collision_prob = 0.0;
	double survivability = 1.0;
	double obstacle_heading = 0.0;

	visualization_msgs::Marker traj_marker;

	while (sim_clock < time_horizon)
	{
	  // Get Velocity Commands
	  sim_cmd_vel = cl->get_velocity_command(sim_goal, vMax);

	  // get navigation function at orig pose
	  nav_fn_t0 = mpepc_sensor_->getGlobalPointPotential(sim_pose);

	  // Update pose
	  current_yaw = current_yaw + (sim_cmd_vel.angular.z * DELTA_SIM_TIME);
	  sim_pose.position.x = sim_pose.position.x + (sim_cmd_vel.linear.x * DELTA_SIM_TIME * cos(current_yaw));
	  sim_pose.position.y = sim_pose.position.y + (sim_cmd_vel.linear.x * DELTA_SIM_TIME * sin(current_yaw));
	  sim_pose.orientation = tf::createQuaternionMsgFromYaw(current_yaw);

	  // Get navigation function at new pose
	  nav_fn_t1 = mpepc_sensor_->getGlobalPointPotential(sim_pose);

	  // Get collision probability
	  if (!collision_detected)
	  {
		  Point obs_pose(-1000.0,-1000.0);
		  double minDist = mpepc_sensor_->min_distance_to_obstacle(sim_pose, &obstacle_heading, &obs_pose);

		  if (minDist <= SAFETY_ZONE)
		  {
			// ROS_INFO("Collision Detected");
			collision_detected = true;
		  }
		  collision_prob = exp(-1*pow(minDist, 2)/SIGMA_DENOM);  // sigma^2
	  }
	  else
	  {
		collision_prob = 1;
	  }


	  // Get survivability
	  survivability = survivability*(1 - collision_prob);

	  expected_collision = expected_collision + ((1-survivability) * C2);

	  // Get progress cost
	  expected_progress = expected_progress + (survivability * (nav_fn_t1 - nav_fn_t0));

	  // Get action cost
	  expected_action = expected_action + (C3 * pow(sim_cmd_vel.linear.x, 2) + C4 * pow(sim_cmd_vel.angular.z, 2))*DELTA_SIM_TIME;

	  // Calculate new EgoPolar coords for goal
	  sim_goal = cl->convert_to_egopolar(sim_pose, current_goal);

	  sim_clock = sim_clock + DELTA_SIM_TIME;
	}

	// Update with angle heuristic - weighted difference between final pose and gradient of navigation function
	double gradient_angle = (nav_fn_t1 - nav_fn_t0);

	expected_progress = expected_progress + C1 * abs(tf::getYaw(sim_pose.orientation) - gradient_angle);

	double sumCost = (expected_collision + expected_progress + expected_action);

	// SUM collision cost, progress cost, action cost
	return sumCost;
}

/**
   * Call back function for nlopt objective function
   * Note: this function is not belong to class
   */
double gps_control::score_trajectory(const std::vector<double> &x, std::vector<double> &grad, void* f_data){
	MpepcMobileRobot * planner = static_cast<MpepcMobileRobot *>(f_data);
	return planner->sim_trajectory(x[0], x[1], x[2], x[3]);
}

void MpepcMobileRobot::find_intermediate_goal_params(gps_agent_pkg::EgoGoal *next_step)
{
	  trajectory_count = 0;

	  int max_iter = 250;  // 30
	  nlopt::opt opt = nlopt::opt(nlopt::GN_DIRECT_NOSCAL, 4);
	  opt.set_min_objective(score_trajectory, this);
	  opt.set_xtol_rel(0.0001);
	  std::vector<double> lb;
	  std::vector<double> rb;
	  lb.push_back(0);
	  lb.push_back(-1.8);
	  lb.push_back(-1.8);
	  lb.push_back(settings_.m_V_MIN);
	  rb.push_back(3.0);
	  rb.push_back(1.8);
	  rb.push_back(1.8);
	  rb.push_back(settings_.m_V_MAX);
	  opt.set_lower_bounds(lb);
	  opt.set_upper_bounds(rb);
	  opt.set_maxeval(max_iter);

	  std::vector<double> k(4);
	  k[0] = 0.0;
	  k[1] = 0.0;
	  k[2] = 0.0;
	  k[3] = 0.0;
	  double minf;

	  opt.optimize(k, minf);

	  ROS_DEBUG("Global Optimization - Trajectories evaluated: %d", trajectory_count);
	  trajectory_count = 0;

	  max_iter = 75;  // 200
	  nlopt::opt opt2 = nlopt::opt(nlopt::LN_BOBYQA, 4);
	  opt2.set_min_objective(score_trajectory, this);
	  opt2.set_xtol_rel(0.0001);
	  std::vector<double> lb2;
	  std::vector<double> rb2;
	  lb2.push_back(0);
	  lb2.push_back(-1.8);
	  lb2.push_back(-3.1);
	  lb2.push_back(settings_.m_V_MIN);
	  rb2.push_back(3.0);
	  rb2.push_back(1.8);
	  rb2.push_back(3.1);
	  rb2.push_back(settings_.m_V_MAX);
	  opt2.set_lower_bounds(lb2);
	  opt2.set_upper_bounds(rb2);
	  opt2.set_maxeval(max_iter);

	  opt2.optimize(k, minf);
	  trajectory_count = 0;

	  next_step->r = k[0];
	  next_step->delta = k[1];
	  next_step->theta = k[2];
	  next_step->vMax = k[3];
	  next_step->k1 = settings_.m_K1;
	  next_step->k2 = settings_.m_K2;

	  return;
}


