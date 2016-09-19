#pragma once

#include "optim/OptimInterfaces.h"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

namespace percepto
{

// Backtracking stepper that reduces the step size until a termination criteria
// is satisfied.
class BacktrackingSearchStepper
: public SearchStepper
{
public:

	typedef std::shared_ptr<BacktrackingSearchStepper> Ptr;

	BacktrackingSearchStepper();

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	void SetInitialStep( double k );
	void SetBacktrackingRatio( double a );
	void SetMaxBacktracks( unsigned int m );
	void SetImprovementRatio( double c );

	virtual void Reset();
	virtual double ComputeStepSize( OptimizationProblem& problem,
	                                const VectorType& direction );

private:

	double _initialStep;
	double _backtrackRatio;
	unsigned int _maxBacktracks;
	double _improvementRatio;

	template <typename InfoType>
	void InitializeFromInfo( const InfoType& info );
};

}