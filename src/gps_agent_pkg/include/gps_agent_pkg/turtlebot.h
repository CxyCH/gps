/*
 * turtlebot.h
 *
 *  Created on: Mar 5, 2017
 *      Author: thobotics
 */

#ifndef GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_TURTLEBOT_H_
#define GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_TURTLEBOT_H_


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
/*
// Controllers.
class TrialController;
// Sensors.
class Sensor;
// Sample.
class Sample;
// Custom ROS messages.
class SampleResult;
class TrialCommand;*/

class Turtlebot: public RobotPlugin {
	/*protected:
	ros::Time last_update_time_;
    // Current trial controller (if any).
    boost::scoped_ptr<TrialController> trial_controller_;
    // Sensor data for the current time step.
    boost::scoped_ptr<Sample> current_time_step_sample_;
    // Sensors.
    std::vector<boost::shared_ptr<Sensor> > sensors_;
    // Subscriber trial commands.
    ros::Subscriber trial_subscriber_;
    // Subscriber for current state report request.
    ros::Subscriber data_request_subscriber_;
    // Publishers.
    // Publish result of a trial, completion of position command, or just a report.
    ros_publisher_ptr(gps_agent_pkg::SampleResult) report_publisher_;
    // Is a trial arm data request pending?
    bool trial_data_request_waiting_;
    // Are the sensors initialized?
    bool sensors_initialized_;
    // Is everything initialized for the trial controller?
    bool controller_initialized_;*/
public:
    // Constructor (this should do nothing).
	Turtlebot();
    // Destructor.
    virtual ~Turtlebot();

    // Init all things needed
    void init(ros::NodeHandle& n);
    // This is the main update function called by the realtime thread when the controller is running.
    void update();
    // This is called by the controller manager before starting the controller.
    void starting();
    // This is called by the controller manager before stopping the controller.
    void stopping();
private:
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
	// TODO: Comment
	//void initialize_sample(boost::scoped_ptr<Sample>& sample, gps::ActuatorType actuator_type);
	// Initialize all of the position controllers.
	void initialize_position_controllers(ros::NodeHandle& n);

	//Helper method to configure all sensors
	//void configure_sensors(OptionsMap &opts);

	// Report publishers
	// Publish a sample with data from up to T timesteps
	//void publish_sample_report(boost::scoped_ptr<Sample>& sample, int T=1);

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
	// Get sensor
	//Sensor *get_sensor(SensorType sensor, gps::ActuatorType actuator_type);
	// This should be empty
	void get_joint_encoder_readings(Eigen::VectorXd &angles, gps::ActuatorType arm) const;
};
}

#endif /* GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_TURTLEBOT_H_ */
