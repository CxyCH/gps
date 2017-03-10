#include <vector>
#include <Eigen/Dense>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <realtime_tools/realtime_publisher.h>

#include "gps_agent_pkg/TrialCommand.h"
#include "gps_agent_pkg/SampleResult.h"
#include "gps_agent_pkg/DataRequest.h"
#include "gps/proto/gps.pb.h"

// Convenience defines.
#define ros_publisher_ptr(X) boost::scoped_ptr<realtime_tools::RealtimePublisher<X> >
#define MAX_TRIAL_LENGTH 2000

// Controllers.
class TrialController;
// Sensors.
class Sensor;
// Sample.
class Sample;
// Custom ROS messages.
class SampleResult;
class TrialCommand;
