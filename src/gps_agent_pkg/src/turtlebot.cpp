/*
 * turtlebot.cpp
 *
 *  Created on: Mar 5, 2017
 *      Author: thobotics
 */
#include "gps_agent_pkg/turtlebot.h"
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/controller.h"
#include "gps_agent_pkg/positioncontroller.h"
#include "gps_agent_pkg/lingausscontroller.h"
#include "gps_agent_pkg/trialcontroller.h"
#include "gps_agent_pkg/LinGaussParams.h"
#include "gps_agent_pkg/ControllerParams.h"
#include "gps_agent_pkg/util.h"
#include "gps/proto/gps.pb.h"
#include <vector>

#ifdef USE_CAFFE
#include "gps_agent_pkg/caffenncontroller.h"
#include "gps_agent_pkg/CaffeParams.h"
#endif

using namespace gps_control;

// Plugin constructor.
Turtlebot::Turtlebot()
{
    // Nothing to do here, since all variables are initialized in initialize(...)
	// Some basic variable initialization.
	controller_counter_ = 0;
	controller_step_length_ = 4;
}

// Destructor.
Turtlebot::~Turtlebot()
{
    // Nothing to do here, since all instance variables are destructed automatically.
}

void Turtlebot::init(ros::NodeHandle& n)
{
	// TODO: Set topic name by parameters
	cmd_pub_ = n.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/navi", 1);
	// TODO: Set this by parameters
	linear_velocities_size_ = 2; // Vx, Vy
	angular_velocities_size_ = 1; // Wz (yaw)

	// Action is linear and angular velocity
	active_arm_torques_.resize(linear_velocities_size_+angular_velocities_size_);

	// Initialize ROS subscribers/publishers, sensors, and navigation controllers.
	initialize(n);
}

// This is called by the controller manager before starting the controller.
void Turtlebot::starting()
{
    // Get current time.
    last_update_time_ = ros::Time::now();
    controller_counter_ = 0;

    // Reset all the sensors. This is important for sensors that try to keep
    // track of the previous state somehow.
    //for (int sensor = 0; sensor < TotalSensorTypes; sensor++)
    /*for (int sensor = 0; sensor < 1; sensor++)
    {
        sensors_[sensor]->reset(this,last_update_time_);
    }*/

    // Reset position controllers.
    nav_controller_->reset(last_update_time_);

    // Reset trial controller, if any.
    if (trial_controller_ != NULL) trial_controller_->reset(last_update_time_);
}

// This is called by the controller manager before stopping the controller.
void Turtlebot::stopping()
{
    // Nothing to do here.
}

void Turtlebot::update()
{
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

    // Setup action and send to robot
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = active_arm_torques_[0];
    /*cmd_vel.linear.y = active_arm_torques_[1];
    cmd_vel.linear.z = active_arm_torques_[2];
    cmd_vel.angular.x = active_arm_torques_[3];
    cmd_vel.angular.y = active_arm_torques_[4];
    cmd_vel.angular.z = active_arm_torques_[5];*/
    cmd_vel.linear.y = active_arm_torques_[1];
    cmd_vel.angular.z = active_arm_torques_[2];
    cmd_pub_.publish(cmd_vel);
}

// Initialize ROS communication infrastructure.
void Turtlebot::initialize_ros(ros::NodeHandle& n)
{
	ROS_INFO_STREAM("Initializing Turtlebot ROS subs/pubs");
	// Create subscribers.
    position_subscriber_ = n.subscribe("/gps_controller_navigation_command", 1, &Turtlebot::nav_subscriber_callback, this);
	trial_subscriber_ = n.subscribe("/gps_controller_trial_command", 1, &Turtlebot::trial_subscriber_callback, this);
	data_request_subscriber_ = n.subscribe("/gps_controller_data_request", 1, &Turtlebot::data_request_subscriber_callback, this);

	// Create publishers.
	report_publisher_.reset(new realtime_tools::RealtimePublisher<gps_agent_pkg::SampleResult>(n, "/gps_controller_report", 1));
}

// Initialize all sensors.
void Turtlebot::initialize_sensors(ros::NodeHandle& n)
{
	ROS_INFO_STREAM("Initializing Tutlebot sensor");
    // Clear out the old sensors.
    sensors_.clear();

    // Create all sensors.
    // NOTE: gps::TRIAL_ARM to only initial sensors_ (get rid of auxiliary sensor)
    // TODO: Create image sensor here.
    int i = MobileSensorType;
	ROS_INFO_STREAM("creating sensor: " + to_string(i));
	boost::shared_ptr<Sensor> sensor(Sensor::create_sensor((SensorType)i,n,this,gps::TRIAL_ARM));
	sensors_.push_back(sensor);

    // Create current state sample and populate it using the sensors.
    current_time_step_sample_.reset(new Sample(MAX_TRIAL_LENGTH));

    // initialize sample
    initialize_sample(current_time_step_sample_, gps::TRIAL_ARM);

    sensors_initialized_ = true;
}

// Initialize position controllers.
void Turtlebot::initialize_position_controllers(ros::NodeHandle& n)
{
	nav_controller_.reset(new NavController(n));
}

// Update the controllers at each time step.
void Turtlebot::update_controllers(ros::Time current_time, bool is_controller_step)
{
	bool trial_init = trial_controller_ != NULL && trial_controller_->is_configured() && controller_initialized_;
	if(!is_controller_step && trial_init){
		return;
	}

	// If we have a trial controller, update that, otherwise update position controller.
	// TODO: Uncomment if implemented navigation controller
	if (trial_init) trial_controller_->update(this, current_time, current_time_step_sample_, active_arm_torques_);
	else nav_controller_->update(this, current_time, current_time_step_sample_, active_arm_torques_);

	// Check if the trial controller finished and delete it.
	if (trial_init && trial_controller_->is_finished()) {

        // Publish sample after trial completion
        publish_sample_report(current_time_step_sample_, trial_controller_->get_trial_length());
        //Clear the trial controller.
        trial_controller_->reset(current_time);
        trial_controller_.reset(NULL);

        // Set the active arm controller to NO_CONTROL.
        // TODO: Uncomment if implemented navigation controller
        /*OptionsMap options;
        options["mode"] = gps::NO_CONTROL;
        active_arm_controller_->configure_controller(options);*/


        // Switch the sensors to run at full frequency.
        for (int sensor = 0; sensor < TotalSensorTypes; sensor++)
        {
            //sensors_[sensor]->set_update(active_arm_controller_->get_update_delay());
        }
	}

	// TODO: Uncomment if implemented navigation controller
	if (nav_controller_->report_waiting){
		if (nav_controller_->is_finished()){
			publish_sample_report(current_time_step_sample_);
			nav_controller_->report_waiting = false;
		}
	}
}

// Position command callback.
void Turtlebot::nav_subscriber_callback(const gps_agent_pkg::NavigationCommand::ConstPtr& msg)
{
	ROS_INFO_STREAM("received navigation command");
	OptionsMap params;

	Eigen::VectorXd position;
	position.resize(msg->position.size());
	for(int i=0; i<position.size(); i++){
		position[i] = msg->position[i];
	}
	params["position"] = position;

	Eigen::VectorXd orientation;

	orientation.resize(msg->quaternion.size());
	for(int i=0; i<orientation.size(); i++){
		orientation[i] = msg->quaternion[i];
	}
	params["orientation"] = orientation;

	nav_controller_->configure_controller(params);
}

// Trial command callback.
void Turtlebot::trial_subscriber_callback(const gps_agent_pkg::TrialCommand::ConstPtr& msg)
{
	RobotPlugin::trial_subscriber_callback(msg);
}

// Trial command callback.
void Turtlebot::data_request_subscriber_callback(const gps_agent_pkg::DataRequest::ConstPtr& msg)
{
	RobotPlugin::data_request_subscriber_callback(msg);
}

// Get current time.
ros::Time Turtlebot::get_current_time() const
{
    return last_update_time_;
}

// Get current encoder readings (robot-dependent).
void Turtlebot::get_joint_encoder_readings(Eigen::VectorXd &angles, gps::ActuatorType arm) const
{
}
