#include "optim/GradientTerminationChecker.h"
#include "argus_utils/utils/ParamUtils.h"

#include <sstream>

namespace argus
{

GradientTerminationChecker::GradientTerminationChecker()
: _minNorm( 1E-3 ) {}

template <typename InfoType>
void GradientTerminationChecker::InitializeFromInfo( const InfoType& info )
{
	double norm;
	if( GetParam( info, "min_norm", norm ) )
	{
		SetMinGradientNorm( norm );
	}
}

void GradientTerminationChecker::Initialize( const ros::NodeHandle& ph )
{
	InitializeFromInfo( ph );
}

void GradientTerminationChecker::Initialize( const YAML::Node& node )
{
	InitializeFromInfo( node );
}

void GradientTerminationChecker::SetMinGradientNorm( double n )
{
	_minNorm = n;
}

std::string GradientTerminationChecker::CheckTermination( OptimizationProblem& problem )
{
	VectorType grad = problem.ComputeGradient();
	double norm = grad.norm();
	if( norm < _minNorm )
	{
		std::stringstream ss;
		ss << "Gradient norm of: " << norm << " less than min: " << _minNorm;
		return ss.str();
	}
	return "";
}

void GradientTerminationChecker::Reset() {}

}