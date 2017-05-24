#pragma once

#include "optim/OptimInterfaces.h"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

namespace argus
{

class GradientSearchDirector
: public SearchDirector
{
public:

	typedef std::shared_ptr<GradientSearchDirector> Ptr;

	GradientSearchDirector();

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	virtual void Reset();
	virtual VectorType ComputeSearchDirection( OptimizationProblem& problem );
};

}