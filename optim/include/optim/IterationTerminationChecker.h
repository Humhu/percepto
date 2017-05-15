#pragma once

#include "optim/OptimInterfaces.h"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

namespace argus
{
	
class IterationTerminationChecker
: public TerminationChecker
{
public:

	typedef std::shared_ptr<IterationTerminationChecker> Ptr;

	IterationTerminationChecker();

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	void SetMaxIterations( unsigned int m );

	virtual std::string CheckTermination( OptimizationProblem& problem );
	virtual void Reset();

private:

	unsigned int _iters;
	unsigned int _maxIters;

	template <typename InfoType>
	void InitializeFromInfo( const InfoType& info );
};

}