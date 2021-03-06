/*
 * mobilerobotsensor.cpp
 *
 *  Created on: Mar 8, 2017
 *      Author: thobotics
 */

#include "gps_agent_pkg/mobilerobotsensor.h"

#include "../include/gps_agent_pkg/mobilerobot.h"

using namespace gps_control;

char* MobileRobotSensor::cost_translation_table_ = NULL;

// Constructor.
MobileRobotSensor::MobileRobotSensor(ros::NodeHandle& n, RobotPlugin *plugin): Sensor(n, plugin)
{
	// Initialize pose.
	position_.resize(3);
	orientation_.resize(4); // Quaternion

	// Initialize velocities
	linear_velocities_.resize(3);
	angular_velocities_.resize(3);

	// Initial position of nearest obstacle.
	nearest_obstacle_.resize(3);

	// TODO: using parameter
	range_data_.resize(30);

	potential_score_.resize(2);

	// Initialize cost map
	tf_ = new tf::TransformListener(ros::Duration(10));
	costmap_ros_ = new costmap_2d::Costmap2DROS("local_costmap", *tf_);
	costmap_ros_->start();

	// For compute obstacle tree
	// NOTE: Copy from costmap_2d_publisher.
	data = NULL;
	obs_tree = NULL;
	if (cost_translation_table_ == NULL)
	{
		cost_translation_table_ = new char[256];

		// special values:
		cost_translation_table_[0] = 0;  // NO obstacle
		cost_translation_table_[253] = 99;  // INSCRIBED obstacle
		cost_translation_table_[254] = 100;  // LETHAL obstacle
		cost_translation_table_[255] = -1;  // UNKNOWN

		// regular cost values scale the range 1 to 252 (inclusive) to fit
		// into 1 to 98 (inclusive).
		for (int i = 1; i < 253; i++)
		{
		  cost_translation_table_[ i ] = char(1 + (97 * (i - 1)) / 251);
		}
	}

	// Initialize subscriber
	n.param<std::string>("/odom_topic", topic_name_, "odom");
	n.param<std::string>("/scan_topic", range_topic_name_, "scan");
	n.param<std::string>("/navfn_topic", potential_topic_name_, "/globalplanner_node/GlobalPlanner/potential");

	subscriber_ = n.subscribe(topic_name_, 1, &MobileRobotSensor::update_data_vector, this);
	range_subscriber_ = n.subscribe(range_topic_name_, 1, &MobileRobotSensor::update_range_data, this);
	navfn_subscriber_ = n.subscribe<nav_msgs::OccupancyGrid> (potential_topic_name_, 1, &MobileRobotSensor::update_navfn, this);
	nearest_obs_pub_ = n.advertise<visualization_msgs::Marker>("nearest_obstacle", 10);

	// Set time.
	previous_pose_time_ = ros::Time(0.0); // This ignores the velocities on the first step.
}

// Destructor.
MobileRobotSensor::~MobileRobotSensor()
{
    // Nothing to do here.
}

// Update the sensor (called every tick).
void MobileRobotSensor::update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step)
{
	if (is_controller_step)
	{
		// Must call this before finding the closest obstacle
		updateObstacleTree(costmap_ros_->getCostmap());

		// TODO: Update current robot pose, velocities
		double update_time = current_time.toSec() - previous_pose_time_.toSec();
		// ...
		// Update stored time.

		// Find the nearest obstacle, default is -1000.0, -1000.0
		double obstacle_heading = 0.0;
		Point obs_pose(-1000.0,-1000.0);
		double minDist;

		{
			boost::mutex::scoped_lock lock(odom_mutex_);
			minDist = min_distance_to_obstacle(cur_pose_, &obstacle_heading, &obs_pose);
			potential_score_[0] = getGlobalPointPotential(cur_pose_);
		}

		nearest_obstacle_[0] = obs_pose.a;
		nearest_obstacle_[1] = obs_pose.b;
		nearest_obstacle_[2] = 0.0;

		visualization_msgs::Marker nearest_points;
		nearest_points.header.frame_id = "odom";
		nearest_points.header.stamp = ros::Time::now();
		nearest_points.ns = "nearest_points";
		nearest_points.action = visualization_msgs::Marker::MODIFY;
		nearest_points.pose.orientation.w = 1;
		nearest_points.id = 0;
		nearest_points.type = visualization_msgs::Marker::POINTS;
		nearest_points.scale.x = 0.2;
		nearest_points.scale.y = 0.2;

		// Points are green
		nearest_points.color.g = 1.0f;
		nearest_points.color.a = 1.0;

		geometry_msgs::Point p;
		p.x = obs_pose.a;
		p.y = obs_pose.b;
		p.z = 1;

		nearest_points.points.push_back(p);
		nearest_obs_pub_.publish(nearest_points);

		ROS_INFO("Cur pose %.2f, %.2f", cur_pose_.position.x, cur_pose_.position.y);
		ROS_INFO("Min distance: %.2f, Pot: %.2f, Pose: %.2f, %.2f", minDist, potential_score_[0], obs_pose.a, obs_pose.b);

		previous_pose_time_ = current_time;
	}
}

void MobileRobotSensor::configure_sensor(OptionsMap &options)
{
	ROS_INFO("configuring mobilerobotsensor");
}

// Set data format and meta data on the provided sample.
void MobileRobotSensor::set_sample_data_format(boost::scoped_ptr<Sample>& sample)
{
    // Set position size and format.
    OptionsMap position_metadata;
    sample->set_meta_data(gps::MOBILE_POSITION,position_.size(),SampleDataFormatEigenVector,position_metadata);

    // Set orientation size and format.
    OptionsMap orientation_metadata;
    sample->set_meta_data(gps::MOBILE_ORIENTATION,orientation_.size(),SampleDataFormatEigenVector,orientation_metadata);

    // Set nearest obstacle size and format.
	OptionsMap nearest_obstacle_metadata;
	sample->set_meta_data(gps::POSITION_NEAREST_OBSTACLE,nearest_obstacle_.size(),SampleDataFormatEigenVector,nearest_obstacle_metadata);

	// Set nearest obstacle size and format.
	OptionsMap potential_score_metadata;
	sample->set_meta_data(gps::POTENTIAL_SCORE,potential_score_.size(),SampleDataFormatEigenVector,potential_score_metadata);

    // Set linear velocities size and format.
	OptionsMap linear_velocities_metadata;
	sample->set_meta_data(gps::MOBILE_VELOCITIES_LINEAR,linear_velocities_.size(),SampleDataFormatEigenVector,linear_velocities_metadata);

	// Set angular velocities size and format.
	OptionsMap angular_velocities_metadata;
	sample->set_meta_data(gps::MOBILE_VELOCITIES_ANGULAR,angular_velocities_.size(),SampleDataFormatEigenVector,angular_velocities_metadata);

	// Set range data size and format.
	OptionsMap range_metadata;
	sample->set_meta_data(gps::MOBILE_RANGE_SENSOR,range_data_.size(),SampleDataFormatEigenVector,range_metadata);


}

// Set data on the provided sample.
void MobileRobotSensor::set_sample_data(boost::scoped_ptr<Sample>& sample, int t)
{
	{
		boost::mutex::scoped_lock lock(odom_mutex_);

		// Set position.
		sample->set_data_vector(t,gps::MOBILE_POSITION,position_.data(),position_.size(),SampleDataFormatEigenVector);

		// Set orientation.
		sample->set_data_vector(t,gps::MOBILE_ORIENTATION,orientation_.data(),orientation_.size(),SampleDataFormatEigenVector);

		// Set linear velocities.
		sample->set_data_vector(t,gps::MOBILE_VELOCITIES_LINEAR,linear_velocities_.data(),linear_velocities_.size(),SampleDataFormatEigenVector);

		// Set angular velocities.
		sample->set_data_vector(t,gps::MOBILE_VELOCITIES_ANGULAR,angular_velocities_.data(),angular_velocities_.size(),SampleDataFormatEigenVector);
	}

	// Set nearest obstacle.
	sample->set_data_vector(t,gps::POSITION_NEAREST_OBSTACLE,nearest_obstacle_.data(),nearest_obstacle_.size(),SampleDataFormatEigenVector);

	sample->set_data_vector(t,gps::POTENTIAL_SCORE,potential_score_.data(),potential_score_.size(),SampleDataFormatEigenVector);

	{
		boost::mutex::scoped_lock lock(range_mutex_);
		// Set range data.
		sample->set_data_vector(t,gps::MOBILE_RANGE_SENSOR,range_data_.data(),range_data_.size(),SampleDataFormatEigenVector);
	}
}

void MobileRobotSensor::update_data_vector(const nav_msgs::Odometry::ConstPtr& msg)
{
	{
		boost::mutex::scoped_lock lock(odom_mutex_);

		// Update current robot pose and velocites
		position_[0] = msg->pose.pose.position.x;
		position_[1] = msg->pose.pose.position.y;
		position_[2] = msg->pose.pose.position.z;

		orientation_[0] = msg->pose.pose.orientation.x;
		orientation_[1] = msg->pose.pose.orientation.y;
		orientation_[2] = msg->pose.pose.orientation.z;
		orientation_[3] = msg->pose.pose.orientation.w;

		linear_velocities_[0] = msg->twist.twist.linear.x;
		linear_velocities_[1] = msg->twist.twist.linear.y;
		linear_velocities_[2] = msg->twist.twist.linear.z;

		angular_velocities_[0] = msg->twist.twist.angular.x;
		angular_velocities_[1] = msg->twist.twist.angular.y;
		angular_velocities_[2] = msg->twist.twist.angular.z;

		cur_pose_.position.x = position_[0];
		cur_pose_.position.y = position_[1];
		cur_pose_.position.z = position_[2];

		cur_pose_.orientation.x = orientation_[0];
		cur_pose_.orientation.y = orientation_[1];
		cur_pose_.orientation.z = orientation_[2];
		cur_pose_.orientation.w = orientation_[3];
	}
}

void MobileRobotSensor::update_range_data(const sensor_msgs::LaserScan::ConstPtr& msg)
{
	{
		boost::mutex::scoped_lock lock(range_mutex_);
		for(int i=0; i < range_data_.size(); i++)
		{
			range_data_[i] = msg->ranges[i];
		}
	}
}

void MobileRobotSensor::update_navfn(const nav_msgs::OccupancyGrid::ConstPtr& nav_cost)
{
	{
	    boost::mutex::scoped_lock lock(global_pot_mutex_);

		global_potarr_ = nav_cost->data;
		global_width_ = nav_cost->info.width;
		global_height_ = nav_cost->info.height;
		origin_x_ = nav_cost->info.origin.position.x;
		origin_y_ = nav_cost->info.origin.position.y;
		resolution_ = nav_cost->info.resolution;
	}
}

geometry_msgs::Point MobileRobotSensor::transformOdomToMap(geometry_msgs::Pose local_pose){
	// Transform to global_pose
	//clock_t begin_time = clock();
	geometry_msgs::PointStamped local_point;
	local_point.header.frame_id = costmap_ros_->getGlobalFrameID();
	// This is important!!!
	// It make transformPose to lookup the latest available transform
	local_point.header.stamp = ros::Time(0);
	local_point.point = local_pose.position;
	geometry_msgs::PointStamped global_point_stamp;
	try{
		tf_->waitForTransform("/map", costmap_ros_->getGlobalFrameID(), ros::Time(0), ros::Duration(10.0));
		tf_->transformPoint("/map", local_point, global_point_stamp);
	}catch (tf::TransformException & ex){
		ROS_ERROR("Transform exception 111 : %s", ex.what());
	}
	//ROS_INFO("Transform plan take %f", float( clock() - begin_time ) /  CLOCKS_PER_SEC);
	return global_point_stamp.point;
}

double MobileRobotSensor::getGlobalPointPotential(geometry_msgs::Pose local_pose){
	//clock_t begin_time = clock();
	// Uninitialized
	if (global_width_ == 0 || global_height_ == 0) return DBL_MAX;
	geometry_msgs::Point currentPoint = transformOdomToMap(local_pose);

	double score;
	{
		boost::mutex::scoped_lock lock(global_pot_mutex_);

		// TODO: Interpolate here
		bool flag = false;
		unsigned int mx, my;
		// Copy from worldToMap from Costmap2D
		if (currentPoint.x < origin_x_ || currentPoint.y < origin_y_)
		  flag = false;
		mx = (int)((currentPoint.x - origin_x_) / resolution_);
		my = (int)((currentPoint.y - origin_y_) / resolution_);
		if (mx < global_width_ && my < global_height_)
		  flag = true;

		//ROS_INFO("Get point potential take %f", float( clock() - begin_time ) /  CLOCKS_PER_SEC);

		if(!flag)
			score = DBL_MAX;

		unsigned int index = my * global_width_ + mx;

		if(global_potarr_[index] == -1){
			score = 100; // Highest
		}else
			score = global_potarr_[index];
	}

	return score;
}

void MobileRobotSensor::updateObstacleTree(costmap_2d::Costmap2D *costmap)
{
	// Create occupancy grid message
	// Copy from costmap_2d_publisher.cpp
	nav_msgs::OccupancyGrid::_info_type::_resolution_type resolution = costmap->getResolution();
	nav_msgs::OccupancyGrid::_info_type::_width_type width = costmap->getSizeInCellsX();
	nav_msgs::OccupancyGrid::_info_type::_height_type height = costmap->getSizeInCellsY();
	double wx, wy;
	costmap->mapToWorld(0, 0, wx, wy);
	nav_msgs::OccupancyGrid::_info_type::_origin_type::_position_type::_x_type x = wx - resolution / 2;
	nav_msgs::OccupancyGrid::_info_type::_origin_type::_position_type::_y_type y = wy - resolution / 2;
	nav_msgs::OccupancyGrid::_info_type::_origin_type::_position_type::_z_type z = 0.0;
	nav_msgs::OccupancyGrid::_info_type::_origin_type::_orientation_type::_w_type w = 1.0;

	nav_msgs::OccupancyGrid::_data_type grid_data;
	grid_data.resize(width * height);

	unsigned char* charData = costmap->getCharMap();
	for (unsigned int i = 0; i < grid_data.size(); i++)
	{
		grid_data[i] = cost_translation_table_[ charData[ i ]];
	}

	// Copy from costmap_translator
	nav_msgs::GridCells obstacles;
	obstacles.cell_width = resolution;
	obstacles.cell_height = resolution;
	for (unsigned int i = 0 ; i < height; ++i)
	{
		for(unsigned int j = 0; j < width; ++j)
		{
		  if(grid_data[i*height+j] == 100)
		  {
			geometry_msgs::Point obstacle_coordinates;
			obstacle_coordinates.x = (j * obstacles.cell_height) + x + (resolution/2.0);
			obstacle_coordinates.y = (i * obstacles.cell_width) + y + (resolution/2.0);
			obstacle_coordinates.z = 0;
			obstacles.cells.push_back(obstacle_coordinates);
		  }
		}
	}

	// Copy from nav_cb
	cost_map = obstacles;

	if(cost_map.cells.size() > 0)
	{
		delete obs_tree;
		delete data;
		data = new flann::Matrix<float>(new float[obstacles.cells.size()*2], obstacles.cells.size(), 2);

		for (size_t i = 0; i < data->rows; ++i)
		{
		  for (size_t j = 0; j < data->cols; ++j)
		  {
			if (j == 0)
			  (*data)[i][j] = cost_map.cells[i].x;
			else
			  (*data)[i][j] = cost_map.cells[i].y;
		  }
		}
		// Obstacle index for fast nearest neighbor search
		obs_tree = new flann::Index<flann::L2<float> >(*data, flann::KDTreeIndexParams(4));
		obs_tree->buildIndex();
	}
}

MinDistResult MobileRobotSensor::find_nearest_neighbor(Point queryPoint)
{
    MinDistResult results;

    flann::Matrix<float> query(new float[2], 1, 2);
    query[0][0] = queryPoint.a;
    query[0][1] = queryPoint.b;

    std::vector< std::vector<int> > indices;
    std::vector< std::vector<float> > dists;

    flann::SearchParams params;
    params.checks = 128;
    params.sorted = true;

    {
      boost::mutex::scoped_lock lock(cost_map_mutex_);
      obs_tree->knnSearch(query, indices, dists, 1, params);
      results.p = Point((*data)[indices[0][0]][0], (*data)[indices[0][0]][1]);
      results.dist = static_cast<double>(dists[0][0]);
    }

    MinDistResult tempResults;
    tempResults.p = Point(cost_map.cells[indices[0][0]].x, cost_map.cells[indices[0][0]].y);

    delete[] query.ptr();
    indices.clear();
    dists.clear();

    return results;
}

double MobileRobotSensor::min_distance_to_obstacle(geometry_msgs::Pose local_current_pose, double *heading, Point *obs_pose)
{
	if(cost_map.cells.size() == 0){
		return 100000;
	}
	// ROS_INFO("In minDist Function");
	Point global(local_current_pose.position.x, local_current_pose.position.y);
	MinDistResult nn_graph_point = find_nearest_neighbor(global);

	double minDist = 100000;
	double head = 0;

	minDist = distance(local_current_pose.position.x, local_current_pose.position.y, nn_graph_point.p.a, nn_graph_point.p.b);

	obs_pose->a = nn_graph_point.p.a;
	obs_pose->b = nn_graph_point.p.b;
	*heading = head;

	return minDist;
}

double MobileRobotSensor::distance(double pose_x, double pose_y, double obx, double oby)
{
  double diffx = obx - pose_x;
  double diffy = oby - pose_y;
  double dist = sqrt(diffx*diffx + diffy*diffy);
  return dist;
}

double MobileRobotSensor::mod(double x, double y)
{
  double m= x - y * floor(x/y);
  // handle boundary cases resulted from floating-point cut off:
  if (y > 0)              // modulo range: [0..y)
  {
	if (m >= y)           // Mod(-1e-16             , 360.    ): m= 360.
	  return 0;

	if (m < 0)
	{
	  if (y+m == y)
		return 0;     // just in case...
	  else
		return y+m;  // Mod(106.81415022205296 , _TWO_PI ): m= -1.421e-14
	}
  }
  else                    // modulo range: (y..0]
  {
	if (m <= y)           // Mod(1e-16              , -360.   ): m= -360.
	  return 0;

	if (m>0 )
	{
	  if (y+m == y)
		return 0;    // just in case...
	  else
		return y+m;  // Mod(-106.81415022205296, -_TWO_PI): m= 1.421e-14
	}
  }

  return m;
}
