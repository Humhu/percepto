#include "optim/RuntimeTerminationChecker.h"
#include "argus_utils/utils/ParamUtils.h"
#include <sstream>

using namespace argus;

namespace percepto
{

RuntimeTerminationChecker::RuntimeTerminationChecker()
: _initialized( false ), _maxTime( 1.0 ) {}

template <typename InfoType>
void RuntimeTerminationChecker::InitializeFromInfo( const InfoType& info )
{
	double rt;
	if( GetParam( info, "max_runtime", rt ) )
	{
		SetMaxRuntime( rt );
	}
}

void RuntimeTerminationChecker::Initialize( const ros::NodeHandle& ph )
{
	InitializeFromInfo( ph );
}

void RuntimeTerminationChecker::Initialize( const YAML::Node& node )
{
	InitializeFromInfo( node );
}

void RuntimeTerminationChecker::SetMaxRuntime( double m )
{
	_maxTime = m;
}

std::string RuntimeTerminationChecker::CheckTermination( OptimizationProblem& problem )
{
	if( !_initialized )
	{
		_startTime = clock();
		_initialized = true;
		return "";
	}

	double runtime = ( (double) clock() - _startTime ) / CLOCKS_PER_SEC;
	if( runtime > _maxTime )
	{
		std::stringstream ss;
		ss << "Runtime of " << runtime << " (s) exceeds max of " << _maxTime << " (s).";
		return ss.str();
	}

	return "";
}

void RuntimeTerminationChecker::Reset()
{
	_initialized = false;
}

}