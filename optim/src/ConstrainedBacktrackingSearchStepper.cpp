#include "optim/ConstrainedBacktrackingSearchStepper.h"
#include "argus_utils/utils/ParamUtils.h"

namespace argus
{

ConstrainedBacktrackingSearchStepper::ConstrainedBacktrackingSearchStepper() 
: _constraintBacktrackRatio( 0.5 ), _maxConstraintBacktracks( 10 )
{}

void ConstrainedBacktrackingSearchStepper::Initialize( const ros::NodeHandle& ph )
{
	BacktrackingSearchStepper::Initialize( ph );
	InitializeFromInfo( ph );
}

void ConstrainedBacktrackingSearchStepper::Initialize( const YAML::Node& node )
{
	BacktrackingSearchStepper::Initialize( node );
	InitializeFromInfo( node );
}

template <typename InfoType>
void ConstrainedBacktrackingSearchStepper::InitializeFromInfo( const InfoType& info )
{
	double a;
	unsigned int m;
	if( GetParam( info, "constraint_backtrack_ratio", a ) )
	{
		SetConstraintBacktrackingRatio( a );
	}
	if( GetParam( info, "max_constraint_backtracks", m ) )
	{
		SetMaxConstraintBacktracks( m );
	}
}

void ConstrainedBacktrackingSearchStepper::SetConstraintBacktrackingRatio( double a )
{
	_constraintBacktrackRatio = a;
}

void ConstrainedBacktrackingSearchStepper::SetMaxConstraintBacktracks( unsigned int m )
{
	_maxConstraintBacktracks = m;
}

void ConstrainedBacktrackingSearchStepper::Reset() {}

double ConstrainedBacktrackingSearchStepper::ComputeStepSize( OptimizationProblem& problem,
                                                              const VectorType& direction )
{
	ConstrainedOptimizationProblem& conProb = 
	    dynamic_cast<ConstrainedOptimizationProblem&>( problem );

	VectorType initialParams = conProb.GetParameters();
	double step = BacktrackingSearchStepper::ComputeStepSize( problem, direction );
	
	for( unsigned int i = 0; i < _maxConstraintBacktracks; ++i )
	{
		conProb.SetParameters( initialParams + step * direction );
		if( conProb.IsSatisfied() ) 
		{
			// std::cout << "Constraints satisfied with step: " << step << ". Breaking..." << std::endl;
			break; 
		}
		step *= _constraintBacktrackRatio;
	}

	conProb.SetParameters( initialParams );
	return step;
}

}