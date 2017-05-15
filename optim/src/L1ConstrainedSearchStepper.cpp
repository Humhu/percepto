#include "optim/L1ConstrainedSearchStepper.h"
#include "argus_utils/utils/ParamUtils.h"

namespace argus
{

L1ConstrainedSearchStepper::L1ConstrainedSearchStepper() 
: _stepSize( 1.0 ), _maxL1( 1.0 ) {}

template <typename InfoType>
void L1ConstrainedSearchStepper::InitializeFromInfo( const InfoType& info )
{
	double m, a;
	if( GetParam( info, "max_l1_norm", m ) )
	{
		SetMaxL1Norm( m );
	}
	if( GetParam( info, "step_size", a ) )
	{
		SetStepSize( a );
	}
}

void L1ConstrainedSearchStepper::Initialize( const ros::NodeHandle& ph )
{
	InitializeFromInfo( ph );
}

void L1ConstrainedSearchStepper::Initialize( const YAML::Node& node )
{
	InitializeFromInfo( node );
}

void L1ConstrainedSearchStepper::SetStepSize( double a )
{
	_stepSize = a;
}

void L1ConstrainedSearchStepper::SetMaxL1Norm( double m )
{
	_maxL1 = m;
}

void L1ConstrainedSearchStepper::Reset() {}

double L1ConstrainedSearchStepper::ComputeStepSize( OptimizationProblem& problem,
                                                    const VectorType& direction )
{
	double maxVal = _stepSize * direction.array().abs().maxCoeff();
	if( maxVal > _maxL1 ) { return _stepSize * _maxL1 / maxVal; }
	return _stepSize;
}

}