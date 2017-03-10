#include "gps_agent_pkg/turtlebot_test.h"
#include <geometry_msgs/Pose.h>

#define POSE "pose"

// Publishers.
// Publish result of a trial, completion of position command, or just a report.
ros_publisher_ptr(gps_agent_pkg::SampleResult) report_publisher_;
ros::Publisher chatter_pub;

/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */
void chatterCallback(const gps_agent_pkg::TrialCommand::ConstPtr& msg)
{
  geometry_msgs::Pose stage_pose;
  stage_pose.position.x = 9.977;
  stage_pose.position.y = 1.739;
  stage_pose.orientation.x = 0.707;
  stage_pose.orientation.y = 0.707;
  chatter_pub.publish(stage_pose);

  ROS_INFO("I heard frequency: [%f]", msg->frequency);
  report_publisher_->unlockAndPublish();
}

void resetStage(){
	// TOOD: Something HERE

}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "turtlebot_test");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("/gps_controller_trial_command", 1, chatterCallback);
  report_publisher_.reset(new realtime_tools::RealtimePublisher<gps_agent_pkg::SampleResult>(n, "/gps_controller_report", 1));
  chatter_pub = n.advertise<geometry_msgs::Pose>("/cmd_pose", 1000);

  ROS_INFO("Initialized");

  ros::spin();

  return 0;
}
