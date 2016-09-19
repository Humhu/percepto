#include "optim/ConstraintTerminationChecker.h"

namespace percepto
{

ConstraintTerminationChecker::ConstraintTerminationChecker() {}

void ConstraintTerminationChecker::Initialize( const ros::NodeHandle& ph )
{}

void ConstraintTerminationChecker::Initialize( const YAML::Node& node )
{}

std::string ConstraintTerminationChecker::CheckTermination( OptimizationProblem& problem )
{
	ConstrainedOptimizationProblem& conProb = 
	    dynamic_cast<ConstrainedOptimizationProblem&>( problem );
	if( conProb.IsSatisfied() )
	{
		return "";
	}
	else
	{
		return "Constraints violated!";
	}
}

void ConstraintTerminationChecker::Reset() 
{}

}