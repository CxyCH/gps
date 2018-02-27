/*
 * globalplanner_node.cpp
 *
 *  Created on: Feb 17, 2018
 *      Author: thobotics
 */
#include <tf/transform_listener.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <global_planner/planner_core.h>
#include <geometry_msgs/PoseStamped.h>

costmap_2d::Costmap2DROS* costmap;
global_planner::GlobalPlanner* planner;
std::vector<geometry_msgs::PoseStamped> global_plan;
geometry_msgs::PoseStamped global_goal;

bool makePlan(const geometry_msgs::PoseStamped& goal, std::vector<geometry_msgs::PoseStamped>& plan){
    boost::unique_lock<costmap_2d::Costmap2D::mutex_t> lock(*(costmap->getCostmap()->getMutex()));

    //make sure to set the plan to be empty initially
    plan.clear();

    //since this gets called on handle activate
    if(costmap == NULL) {
      ROS_ERROR("Planner costmap ROS is NULL, unable to create global plan");
      return false;
    }

    //get the starting pose of the robot
    tf::Stamped<tf::Pose> global_pose;
    if(!costmap->getRobotPose(global_pose)) {
      ROS_WARN("Unable to get starting pose of robot, unable to create global plan");
      return false;
    }

    geometry_msgs::PoseStamped start;
    tf::poseStampedTFToMsg(global_pose, start);

    //if the planner fails or returns a zero length plan, planning failed
    if(!planner->makePlan(goal, start, plan) || plan.empty()){
      ROS_DEBUG_NAMED("move_base","Failed to find a  plan to point (%.2f, %.2f)", goal.pose.position.x, goal.pose.position.y);
      return false;
    }

    // Reverse the plan
    geometry_msgs::PoseStamped goal_copy = goal;
    goal_copy.header.stamp = goal.header.stamp;
    plan[0] = goal_copy;
		std::reverse(plan.begin(), plan.end());

    return true;
}

void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& goal)
{
    ROS_INFO("Goal: [%f, %f]", goal->pose.position.x, goal->pose.position.y);
    global_goal.header = goal->header;
    global_goal.pose = goal->pose;

	if(global_goal.header.frame_id != "" && makePlan(global_goal, global_plan)){
		ROS_INFO("Planned");
	}
}

int main(int argc, char** argv){
    ros::init(argc, argv, "globalplanner_node");
    ros::NodeHandle n;
    tf::TransformListener tf(ros::Duration(10));

    costmap = new costmap_2d::Costmap2DROS("global_costmap", tf);
	planner = new global_planner::GlobalPlanner();
	planner->initialize("GlobalPlanner", costmap);

	ros::Subscriber sub = n.subscribe("global_goal", 1, goalCallback);

	ros::spin();

    return 0;
}
