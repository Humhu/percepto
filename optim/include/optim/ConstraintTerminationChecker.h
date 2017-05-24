#pragma once

#include "optim/ConstrainedBacktrackingSearchStepper.h"
#include "optim/OptimInterfaces.h"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

namespace argus
{

class ConstraintTerminationChecker
: public TerminationChecker
{
public:

	typedef std::shared_ptr<ConstraintTerminationChecker> Ptr;

	ConstraintTerminationChecker();

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	virtual std::string CheckTermination( OptimizationProblem& problem );
	virtual void Reset();
};

}