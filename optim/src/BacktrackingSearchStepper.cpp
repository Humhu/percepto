#include "optim/BacktrackingSearchStepper.h"
#include "argus_utils/utils/ParamUtils.h"

namespace argus
{

BacktrackingSearchStepper::BacktrackingSearchStepper()
: _initialStep( 1.0 ), _backtrackRatio( 0.5 ), _maxBacktracks( 10 ), _improvementRatio( 0.5 ) {}

void BacktrackingSearchStepper::Initialize( const ros::NodeHandle& ph )
{
	InitializeFromInfo( ph );
}

void BacktrackingSearchStepper::Initialize( const YAML::Node& node )
{
	InitializeFromInfo( node );
}

template <typename InfoType>
void BacktrackingSearchStepper::InitializeFromInfo( const InfoType& info )
{
	double k, a, c;
	unsigned int m;
	if( GetParam( info, "initial_step", k ) )
	{
		SetInitialStep( k );
	}
	if( GetParam( info, "backtrack_ratio", a ) )
	{
		SetBacktrackingRatio( a );
	}
	if( GetParam( info, "max_backtracks", m ) )
	{
		SetMaxBacktracks( m );
	}
	if( GetParam( info, "improvement_ratio", c ) )
	{
		SetImprovementRatio( c );
	}
}

void BacktrackingSearchStepper::SetInitialStep( double k )
{
	_initialStep = k;
}

void BacktrackingSearchStepper::SetBacktrackingRatio( double a )
{
	_backtrackRatio = a;
}

void BacktrackingSearchStepper::SetMaxBacktracks( unsigned int m )
{
	_maxBacktracks = m;
}

void BacktrackingSearchStepper::SetImprovementRatio( double c )
{
	_improvementRatio = c;
}

void BacktrackingSearchStepper::Reset() {}

double BacktrackingSearchStepper::ComputeStepSize( OptimizationProblem& problem,
                                                   const VectorType& direction )
{
	VectorType initialParams = problem.GetParameters();
	double step = _initialStep;
	
	double initialObjective = problem.ComputeObjective();
	VectorType gradient = problem.ComputeGradient();
	double m = gradient.dot( direction ) / direction.norm();

	for( unsigned int i = 0; i < _maxBacktracks; ++i )
	{
		problem.SetParameters( initialParams + step * direction );
		double currentObjective = problem.ComputeObjective();
		double deltaObjective = currentObjective - initialObjective;

		if( problem.IsMinimization() && 
		    deltaObjective < m * step * _improvementRatio ) { break; }
		else if( !problem.IsMinimization() &&
		         deltaObjective > m * step * _improvementRatio ) { break; }

		step *= _backtrackRatio;
	}

	problem.SetParameters( initialParams );
	return step;
}

}