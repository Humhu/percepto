#pragma once

#include "optim/BacktrackingSearchStepper.h"

namespace argus
{

class ConstrainedOptimizationProblem
: virtual public OptimizationProblem
{
public:

	ConstrainedOptimizationProblem() {}
	virtual ~ConstrainedOptimizationProblem() {}

	virtual bool IsSatisfied() = 0;
};

class ConstrainedBacktrackingSearchStepper
: public BacktrackingSearchStepper
{
public:

	typedef std::shared_ptr<ConstrainedBacktrackingSearchStepper> Ptr;

	ConstrainedBacktrackingSearchStepper();

	void SetConstraintBacktrackingRatio( double a );
	void SetMaxConstraintBacktracks( unsigned int m );

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	virtual double ComputeStepSize( OptimizationProblem& problem,
	                                const VectorType& direction );
	virtual void Reset();

private:

	double _constraintBacktrackRatio;
	unsigned int _maxConstraintBacktracks;

	template <typename TypeInfo>
	void InitializeFromInfo( const TypeInfo& info );
};

}