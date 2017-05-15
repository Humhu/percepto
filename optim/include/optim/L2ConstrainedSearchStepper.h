#pragma once

#include "optim/OptimInterfaces.h"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

namespace argus
{

class L2ConstrainedSearchStepper
: public SearchStepper
{
public:

	typedef std::shared_ptr<L2ConstrainedSearchStepper> Ptr;

	L2ConstrainedSearchStepper();

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	void SetStepSize( double a );
	void SetMaxL2Norm( double m );

	virtual void Reset();
	virtual double ComputeStepSize( OptimizationProblem& problem,
	                                const VectorType& direction );

private:

	double _stepSize;
	double _maxL2;

	template <typename InfoType>
	void InitializeFromInfo( const InfoType& info );

};

}