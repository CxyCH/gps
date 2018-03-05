/*
 * mobilerobotsensor.h
 *
 *  Created on: Mar 8, 2017
 *      Author: thobotics
 */

#ifndef GPS_AGENT_PKG_INCLUDE_MOBILEROBOTSENSOR_H_
#define GPS_AGENT_PKG_INCLUDE_MOBILEROBOTSENSOR_H_

#include "gps/proto/gps.pb.h"

// Superclass.
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/sample.h"

#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/costmap_2d.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/GridCells.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>
#include <sensor_msgs/LaserScan.h>
#include <visualization_msgs/Marker.h>
#include <vector>
#include "flann/flann.hpp"

// This sensor writes to the following data types:
// MOBILE_POSITION
// MOBILE_ORIENTATION
// MOBILE_VELOCITIES_LINEAR
// MOBILE_VELOCITIES_ANGULAR
// POSITION_NEAREST_OBSTACLE

using namespace std;

namespace gps_control
{
static const double PI= 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348;
static const double TWO_PI= 6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696;
static const double minusPI= -3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348;

struct Point {
  float a;
  float b;
  int member;
  int p_idx;
  Point(float x, float y) : a(x), b(y), member(-1), p_idx(0) {}
  Point() : a(0), b(0), member(-1), p_idx(0) {}
  inline bool operator==(Point p) {
	 if (p.a == a && p.b == b)
		return true;
	 else
		return false;
  }
};

struct MinDistResult {
  Point p;
  double dist;
};

class MobileRobotSensor: public Sensor
{
private:
	// TODO: Does it need to change to ROS style
	// Pose
	Eigen::VectorXd position_;
	Eigen::VectorXd orientation_;
	// Velocities
	Eigen::VectorXd linear_velocities_;
	Eigen::VectorXd angular_velocities_;
	// Position to nearest obstacle
	Eigen::VectorXd nearest_obstacle_;
	// Array of range sensor data
	Eigen::VectorXd range_data_;
	// Current global potential point
	Eigen::VectorXd potential_score_;
	// Subscribers
	ros::Subscriber subscriber_;
	ros::Subscriber range_subscriber_;
	ros::Subscriber navfn_subscriber_;
	// Publishers
	ros::Publisher nearest_obs_pub_;
	std::string topic_name_;
	std::string range_topic_name_;
	std::string potential_topic_name_;

	// Time from last update when the previous pose were recorded (necessary to compute velocities).
	ros::Time previous_pose_time_;

	tf::TransformListener* tf_;
	costmap_2d::Costmap2DROS* costmap_ros_;
	boost::mutex odom_mutex_, range_mutex_;
	geometry_msgs::Pose cur_pose_;

	// Building obstacle tree
	boost::mutex cost_map_mutex_;
	static char* cost_translation_table_;
	nav_msgs::GridCells cost_map;
	flann::Index<flann::L2<float> > * obs_tree;
	flann::Matrix<float> * data;

	// Global Navigation Function
	boost::mutex global_pot_mutex_;
	std::vector<int8_t> global_potarr_;
	unsigned int global_width_ = 0, global_height_ = 0;
	double origin_x_ = 0, origin_y_ = 0, resolution_ = 0;

	// Auxiliary function
	double mod(double x, double y);
	double distance(double pose_x, double pose_y, double obx, double oby);
	void updateObstacleTree(costmap_2d::Costmap2D *costmap);
	geometry_msgs::Point transformOdomToMap(geometry_msgs::Pose local_pose);
	MinDistResult find_nearest_neighbor(Point queryPoint);

	// Subscriber topic
	void update_data_vector(const nav_msgs::Odometry::ConstPtr& msg);
	void update_range_data(const sensor_msgs::LaserScan::ConstPtr& msg);
	void update_navfn(const nav_msgs::OccupancyGrid::ConstPtr& msg);

public:
	// Constructor.
	MobileRobotSensor(ros::NodeHandle& n, RobotPlugin *plugin);
	// Destructor.
	virtual ~MobileRobotSensor();
	// Update the sensor (called every tick).
	virtual void update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step);
	// Configure the sensor (for sensor-specific trial settings).
	virtual void configure_sensor(OptionsMap &options);
	// Set data format and meta data on the provided sample.
	virtual void set_sample_data_format(boost::scoped_ptr<Sample>& sample);
	// Set data on the provided sample.
	virtual void set_sample_data(boost::scoped_ptr<Sample>& sample, int t);

	geometry_msgs::Pose getCurrentRobotPose();
	double getGlobalPointPotential(geometry_msgs::Pose local_pose);
	double min_distance_to_obstacle(geometry_msgs::Pose local_current_pose, double *heading, Point *obs_pose);
};
}


#endif /* GPS_AGENT_PKG_INCLUDE_MOBILEROBOTSENSOR_H_ */
