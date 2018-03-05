/*
 * mpepcmobilerobot.h
 *
 *  Created on: Feb 20, 2018
 *      Author: thobotics
 */

#ifndef GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_MPEPCMOBILEROBOT_H_
#define GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_MPEPCMOBILEROBOT_H_


#include <vector>
#include <ros/ros.h>
#include <nlopt.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <gps_agent_pkg/EgoGoal.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "gps_agent_pkg/mobilerobot.h"
#include "gps_agent_pkg/mobilerobotsensor.h"
#include "gps_agent_pkg/mpepccontrollaw.h"
#include "gps/proto/gps.pb.h"

namespace gps_control
{
double score_trajectory(const std::vector<double> &x, std::vector<double> &grad, void* f_data);

class MpepcMobileRobot: public MobileRobot {
public:
    // Constructor (this should do nothing).
	MpepcMobileRobot();
    // Destructor.
    virtual ~MpepcMobileRobot();

    // Init all things needed
    void init(ros::NodeHandle& n, bool use_mpepc);
    // This is the main update function called by the realtime thread when the controller is running.
    void update();

    // This function is used by the optimizer to score different trajectories
    double sim_trajectory(double r, double delta, double theta, double vMax);
private:
	// Action publisher
    gps_agent_pkg::EgoGoal ego_goal_;
    ros::Publisher ego_goal_pub_;

	// NLOPT
	bool use_mpepc_;
	MobileRobotSensor *mpepc_sensor_;
	geometry_msgs::Pose sim_current_pose_;
	mpepc_local_planner::ControlLawSettings settings_;
	mpepc_local_planner::ControlLaw* cl;

	// Trajectory publish
	int trajectory_count;
	ros::Publisher l_plan_pub_;

	// Trajectory Optimization Params
	double TIME_HORIZON;
	double DELTA_SIM_TIME;
	double SAFETY_ZONE;

	// Cost function params
	double C1;
	double C2;
	double C3;
	double C4;
	double SIGMA;

	// Use NLOPT to find the next subgoal for the trajectory generator
	void find_intermediate_goal_params(gps_agent_pkg::EgoGoal *next_step);
	geometry_msgs::PoseArray get_trajectory_viz(gps_agent_pkg::EgoGoal new_coords);
};
}

#endif /* GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_MPEPCMOBILEROBOT_H_ */
