#include "optim/IterationTerminationChecker.h"
#include "argus_utils/utils/ParamUtils.h"
#include <sstream>

namespace argus
{

IterationTerminationChecker::IterationTerminationChecker()
: _iters( 0 ), _maxIters( 100 ) {}

template <typename InfoType>
void IterationTerminationChecker::InitializeFromInfo( const InfoType& info )
{
	unsigned int maxIters;
	if( GetParam( info, "max_iterations", maxIters ) )
	{
		SetMaxIterations( maxIters );
	}
}

void IterationTerminationChecker::Initialize( const ros::NodeHandle& ph )
{
	InitializeFromInfo( ph );
}

void IterationTerminationChecker::Initialize( const YAML::Node& node )
{
	InitializeFromInfo( node );
}

void IterationTerminationChecker::SetMaxIterations( unsigned int m )
{
	_maxIters = m;
}

std::string IterationTerminationChecker::CheckTermination( OptimizationProblem& problem )
{
	_iters++;
	if( _iters > _maxIters )
	{
		std::stringstream ss;
		ss << "Iteration " << _iters << " exceeds max " << _maxIters;
		return ss.str();
	}
	return "";
}

void IterationTerminationChecker::Reset()
{
	_iters = 0;
}

}