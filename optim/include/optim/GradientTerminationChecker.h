#pragma once

#include "optim/OptimInterfaces.h"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

namespace argus
{

class GradientTerminationChecker
: public TerminationChecker
{
public:

	typedef std::shared_ptr<GradientTerminationChecker> Ptr;

	GradientTerminationChecker();

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	void SetMinGradientNorm( double n );

	virtual std::string CheckTermination( OptimizationProblem& problem );
	virtual void Reset();

private:

	double _minNorm;

	template <typename InfoType>
	void InitializeFromInfo( const InfoType& info );

};

}