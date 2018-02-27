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
#include <gps_agent_pkg/EgoGoal.h>
#include "gps_agent_pkg/mobilerobot.h"
#include "gps/proto/gps.pb.h"

namespace gps_control
{

class MpepcMobileRobot: public MobileRobot {
public:
    // Constructor (this should do nothing).
	MpepcMobileRobot();
    // Destructor.
    virtual ~MpepcMobileRobot();

    // Init all things needed
    void init(ros::NodeHandle& n);
    // This is the main update function called by the realtime thread when the controller is running.
    void update();
private:
	// Action publisher
	ros::Publisher ego_goal_pub_;
};
}

#endif /* GPS_AGENT_PKG_INCLUDE_GPS_AGENT_PKG_MPEPCMOBILEROBOT_H_ */
