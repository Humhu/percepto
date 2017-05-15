#include "optim/FixedSearchStepper.h"
#include "argus_utils/utils/ParamUtils.h"

namespace argus
{

FixedSearchStepper::FixedSearchStepper() 
: _stepSize( 1.0 ) {}

template <typename InfoType>
void FixedSearchStepper::InitializeFromInfo( const InfoType& info )
{
	double alpha;
	if( GetParam( info, "step_size", alpha ) )
	{
		SetStepSize( alpha );
	}
}

void FixedSearchStepper::Initialize( const ros::NodeHandle& ph )
{
	InitializeFromInfo( ph );
}

void FixedSearchStepper::Initialize( const YAML::Node& node )
{
	InitializeFromInfo( node );
}

void FixedSearchStepper::SetStepSize( double s )
{
	_stepSize = s;
}

void FixedSearchStepper::Reset() {}

double FixedSearchStepper::ComputeStepSize( OptimizationProblem& problem,
                                            const VectorType& direction )
{
	return _stepSize;
}

}