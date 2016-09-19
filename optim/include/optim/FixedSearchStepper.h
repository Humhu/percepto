#pragma once

#include "optim/OptimInterfaces.h"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

namespace percepto
{

class FixedSearchStepper
: public SearchStepper
{
public:

	typedef std::shared_ptr<FixedSearchStepper> Ptr;

	FixedSearchStepper();

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	void SetStepSize( double s );

	virtual void Reset();
	virtual double ComputeStepSize( OptimizationProblem& problem,
	                                const VectorType& direction );

private:

	double _stepSize;

	template <typename InfoType>
	void InitializeFromInfo( const InfoType& info );
};

}