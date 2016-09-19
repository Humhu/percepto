#pragma once

#include "optim/OptimInterfaces.h"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

#include <ctime>

namespace percepto
{

class RuntimeTerminationChecker
: public TerminationChecker
{
public:

	typedef std::shared_ptr<RuntimeTerminationChecker> Ptr;

	RuntimeTerminationChecker();

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	void SetMaxRuntime( double m );

	virtual std::string CheckTermination( OptimizationProblem& problem );
	virtual void Reset();

private:

	bool _initialized;
	clock_t _startTime;
	double _maxTime;

	template <typename InfoType>
	void InitializeFromInfo( const InfoType& info );

};

}