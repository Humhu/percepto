#pragma once

#include "optim/OptimInterfaces.h"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

namespace argus
{

class L1ConstrainedSearchStepper
: public SearchStepper
{
public:

	typedef std::shared_ptr<L1ConstrainedSearchStepper> Ptr;

	L1ConstrainedSearchStepper();

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	void SetStepSize( double a );
	void SetMaxL1Norm( double m );

	virtual void Reset();
	virtual double ComputeStepSize( OptimizationProblem& problem,
	                                const VectorType& direction );

private:

	double _stepSize;
	double _maxL1;

	template <typename InfoType>
	void InitializeFromInfo( const InfoType& info );

};

}