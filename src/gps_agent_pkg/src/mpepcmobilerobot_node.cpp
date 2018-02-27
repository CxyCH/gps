/*
 * mobilerobot_node.cpp
 *
 *  Created on: Mar 5, 2017
 *      Author: thobotics
 */
#include "gps_agent_pkg/mpepcmobilerobot.h"
using namespace gps_control;

int main(int argc, char** argv){
  ros::init(argc, argv, "mpepcmobilerobot_node");
  ros::NodeHandle n;

  // NOTE: Rate depend on simulator time
  // E.g: stage_ros real-time is 10Hz (/clock topic)
  ros::Rate rate(20);

  MpepcMobileRobot mobilerobot;
  mobilerobot.init(n);
  mobilerobot.starting();

  // Control Loop and code
  while (ros::ok())
  {
	  mobilerobot.update();
	  ros::spinOnce();
	  rate.sleep();
  }

  // It should never be called
  mobilerobot.stopping();

  return(0);
}



