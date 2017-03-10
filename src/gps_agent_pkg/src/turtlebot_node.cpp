/*
 * turtlebot_node.cpp
 *
 *  Created on: Mar 5, 2017
 *      Author: thobotics
 */
#include "gps_agent_pkg/turtlebot.h"
using namespace gps_control;

int main(int argc, char** argv){
  ros::init(argc, argv, "turtlebot_node");
  ros::NodeHandle n;

  // NOTE: Rate depend on simulator time
  // E.g: stage_ros real-time is 10Hz (/clock topic)
  ros::Rate rate(20);

  Turtlebot turtlebot;
  turtlebot.init(n);
  turtlebot.starting();

  // Control Loop and code
  while (ros::ok())
  {
	  turtlebot.update();
	  ros::spinOnce();
	  rate.sleep();
  }

  // It should never be called
  turtlebot.stopping();

  return(0);
}



