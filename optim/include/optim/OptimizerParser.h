#include "optim/ModularOptimizer.h"
#include <yaml-cpp/yaml.h>
#include <ros/ros.h>

namespace percepto
{

ModularOptimizer::Ptr parse_modular_optimizer( const ros::NodeHandle& ph );
ModularOptimizer::Ptr parse_modular_optimizer( const YAML::Node& node );

}