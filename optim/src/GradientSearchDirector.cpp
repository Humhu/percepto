#include "optim/GradientSearchDirector.h"

namespace argus
{

GradientSearchDirector::GradientSearchDirector() {}

void GradientSearchDirector::Initialize( const ros::NodeHandle& ph ) {}

void GradientSearchDirector::Initialize( const YAML::Node& node ) {}

void GradientSearchDirector::Reset() {}

VectorType GradientSearchDirector::ComputeSearchDirection( OptimizationProblem& problem )
{
	if( problem.IsMinimization() ) 
	{
		return -problem.ComputeGradient();
	}
	else
	{
		return problem.ComputeGradient();
	}

}

}