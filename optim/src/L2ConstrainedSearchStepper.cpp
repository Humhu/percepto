#include "optim/L2ConstrainedSearchStepper.h"
#include "argus_utils/utils/ParamUtils.h"

using namespace argus;

namespace percepto
{

L2ConstrainedSearchStepper::L2ConstrainedSearchStepper() 
: _stepSize( 1.0 ), _maxL2( 1.0 ) {}

template <typename InfoType>
void L2ConstrainedSearchStepper::InitializeFromInfo( const InfoType& info )
{
	double m, a;
	if( GetParam( info, "max_l2_norm", m ) )
	{
		SetMaxL2Norm( m );
	}
	if( GetParam( info, "step_size", a ) )
	{
		SetStepSize( a );
	}
}

void L2ConstrainedSearchStepper::Initialize( const ros::NodeHandle& ph )
{
	InitializeFromInfo( ph );
}

void L2ConstrainedSearchStepper::Initialize( const YAML::Node& node )
{
	InitializeFromInfo( node );
}

void L2ConstrainedSearchStepper::SetStepSize( double a )
{
	_stepSize = a;
}

void L2ConstrainedSearchStepper::SetMaxL2Norm( double m )
{
	_maxL2 = m;
}

void L2ConstrainedSearchStepper::Reset() {}

double L2ConstrainedSearchStepper::ComputeStepSize( OptimizationProblem& problem,
                                                    const VectorType& direction )
{
	double norm = ( _stepSize * direction ).norm();
	if( norm > _maxL2 ) { return _stepSize * norm; }
	return _stepSize;
}

}