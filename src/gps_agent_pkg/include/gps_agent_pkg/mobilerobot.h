/*
 * mobilerobot.h
 *
 *  Created on: Mar 5, 2017
 *      Author: thobotics
 */

#ifndef GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_MOBILEROBOT_H_
#define GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_MOBILEROBOT_H_


#include <vector>
#include <Eigen/Dense>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <realtime_tools/realtime_publisher.h>
#include <geometry_msgs/Twist.h>

#include "gps_agent_pkg/NavigationCommand.h"
#include "gps_agent_pkg/TrialCommand.h"
#include "gps_agent_pkg/SampleResult.h"
#include "gps_agent_pkg/DataRequest.h"
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/controller.h"
#include "gps_agent_pkg/navcontroller.h"
#include "gps_agent_pkg/robotplugin.h"
#include "gps/proto/gps.pb.h"

// Convenience defines.
#define ros_publisher_ptr(X) boost::scoped_ptr<realtime_tools::RealtimePublisher<X> >
#define MAX_TRIAL_LENGTH 2000

namespace gps_control
{

class MobileRobot: public RobotPlugin {
public:
    // Constructor (this should do nothing).
	MobileRobot();
    // Destructor.
    virtual ~MobileRobot();

    // Init all things needed
    void init(ros::NodeHandle& n);
    // This is the main update function called by the realtime thread when the controller is running.
    void update();
    // This is called by the controller manager before starting the controller.
    void starting();
    // This is called by the controller manager before stopping the controller.
    void stopping();
protected:
    // Counter for keeping track of controller steps.
	int controller_counter_;
	// Length of controller steps in ms.
	int controller_step_length_;
    // Position controller for passive arm.
	boost::scoped_ptr<NavController> nav_controller_;
	// Linear velocities size
	int linear_velocities_size_;
	// Angular velocities size
	int angular_velocities_size_;
	// Action publisher
	ros::Publisher cmd_pub_;

	// Initialize all of the ROS subscribers and publishers.
	void initialize_ros(ros::NodeHandle& n);
    // Initialize all of the sensors (this also includes FK computation objects).
	void initialize_sensors(ros::NodeHandle& n);
	// Initialize all of the position controllers.
	void initialize_position_controllers(ros::NodeHandle& n);

	// Subscriber callbacks.
    // Position command callback.
    void nav_subscriber_callback(const gps_agent_pkg::NavigationCommand::ConstPtr& msg);
	// Trial command callback.
	void trial_subscriber_callback(const gps_agent_pkg::TrialCommand::ConstPtr& msg);
	// Data request callback.
	void data_request_subscriber_callback(const gps_agent_pkg::DataRequest::ConstPtr& msg);

	// Update functions.
	// Update the sensors at each time step.
	//void update_sensors(ros::Time current_time, bool is_controller_step);
	// Update the controllers at each time step.
	void update_controllers(ros::Time current_time, bool is_controller_step);
	// Accessors.
	// Get current time.
	virtual ros::Time get_current_time() const;
	// This should be empty
	void get_joint_encoder_readings(Eigen::VectorXd &angles, gps::ActuatorType arm) const;
};
}

#endif /* GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_MOBILEROBOT_H_ */
