/*
 * navcontroller.h
 *
 *  Created on: Mar 8, 2017
 *      Author: thobotics
 */

#ifndef GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_NAVCONTROLLER_H_
#define GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_NAVCONTROLLER_H_

// Headers.
#include <Eigen/Dense>

// Superclass.
#include "gps_agent_pkg/controller.h"
#include "gps/proto/gps.pb.h"
#include <geometry_msgs/Pose.h>

namespace gps_control
{

class NavController : public Controller
{
private:
	// Pose
	Eigen::VectorXd position_;
	Eigen::VectorXd orientation_;
	// Agent move_base publisher
	ros::Publisher nav_pub_;
	// TODO: Change this by checking error
	bool finished_;
public:
	// Constructor.
	NavController(ros::NodeHandle& n);
	// Destructor.
	virtual ~NavController();
	// Update the controller (take an action).
	virtual void update(RobotPlugin *plugin, ros::Time current_time, boost::scoped_ptr<Sample>& sample, Eigen::VectorXd &torques);
	// Configure the controller.
	virtual void configure_controller(OptionsMap &options);
	// Check if controller is finished with its current task.
	virtual bool is_finished() const;
	// Reset the controller -- this is typically called when the controller is turned on.
	virtual void reset(ros::Time update_time);
	// Should this report when position achieved?
	bool report_waiting;
};
}



#endif /* GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_NAVCONTROLLER_H_ */
