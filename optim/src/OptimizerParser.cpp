#include "optim/OptimizerParser.h"
#include "optim/Optimizers.h"
#include "argus_utils/utils/ParamUtils.h"

using namespace argus;

namespace percepto
{

template <typename Info>
ModularOptimizer::Ptr ParseModularOptimizer( const Info& info )
{
	ModularOptimizer::Ptr optimizer = std::make_shared<ModularOptimizer>();

	// Parse search director
	std::string directorType;
	YAML::Node dirInfo;
	GetParamRequired( info, "director", dirInfo );
	GetParamRequired( dirInfo, "type", directorType );
	if( directorType == "gradient" )
	{
		GradientSearchDirector::Ptr dir = std::make_shared<GradientSearchDirector>();
		dir->Initialize( dirInfo );
		optimizer->SetSearchDirector( dir );
	}
	else if( directorType == "adam" )
	{
		AdamSearchDirector::Ptr dir = std::make_shared<AdamSearchDirector>();
		dir->Initialize( dirInfo );
		optimizer->SetSearchDirector( dir );
	}
	else if( directorType == "natural" )
	{
		NaturalSearchDirector::Ptr dir = std::make_shared<NaturalSearchDirector>();
		dir->Initialize( dirInfo );
		optimizer->SetSearchDirector( dir );
	}
	else
	{
		throw std::invalid_argument( "Unknown director type: " + directorType );
	}

	// Parse search stepper
	YAML::Node stepInfo;
	GetParamRequired( info, "stepper", stepInfo );
	std::string stepperType;
	GetParamRequired( stepInfo, "type", stepperType );
	if( stepperType == "fixed" )
	{
		FixedSearchStepper::Ptr step = std::make_shared<FixedSearchStepper>();
		step->Initialize( stepInfo );
		optimizer->SetSearchStepper( step );
	}
	else if( stepperType == "l1_constrained" )
	{
		L1ConstrainedSearchStepper::Ptr step = std::make_shared<L1ConstrainedSearchStepper>();
		step->Initialize( stepInfo );
		optimizer->SetSearchStepper( step );
	}
	else if( stepperType == "l2_constrained" )
	{
		L2ConstrainedSearchStepper::Ptr step = std::make_shared<L2ConstrainedSearchStepper>();
		step->Initialize( stepInfo );
		optimizer->SetSearchStepper( step );
	}
	else if( stepperType == "backtracking" )
	{
		BacktrackingSearchStepper::Ptr step = std::make_shared<BacktrackingSearchStepper>();
		step->Initialize( stepInfo );
		optimizer->SetSearchStepper( step );
	}
	else if( stepperType == "constrained_backtracking" )
	{
		ConstrainedBacktrackingSearchStepper::Ptr step = std::make_shared<ConstrainedBacktrackingSearchStepper>();
		step->Initialize( stepInfo );
		optimizer->SetSearchStepper( step );
	}
	else
	{
		throw std::invalid_argument( "Unknown stepper type: " + stepperType );
	}

	// Parse termination checker
	YAML::Node termInfo;
	GetParamRequired( info, "termination_checker", termInfo );
	YAML::const_iterator iter;
	for( iter = termInfo.begin(); iter != termInfo.end(); ++iter )
	{
		std::string type = iter->first.as<std::string>();
		const YAML::Node& info = iter->second;
		if( type == "iterations" )
		{
			IterationTerminationChecker::Ptr crit = std::make_shared<IterationTerminationChecker>();
			crit->Initialize( info );
			optimizer->AddTerminationChecker( crit );
		}
		else if( type == "runtime" )
		{
			RuntimeTerminationChecker::Ptr crit = std::make_shared<RuntimeTerminationChecker>();
			crit->Initialize( info );
			optimizer->AddTerminationChecker( crit );
		}
		else if( type == "gradient_norm" )
		{
			GradientTerminationChecker::Ptr crit = std::make_shared<GradientTerminationChecker>();
			crit->Initialize( info );
			optimizer->AddTerminationChecker( crit );
		}
		else if( type == "constraints" )
		{
			ConstraintTerminationChecker::Ptr crit = std::make_shared<ConstraintTerminationChecker>();
			crit->Initialize( info );
			optimizer->AddTerminationChecker( crit );
		}
		else
		{
			throw std::invalid_argument( "Unknown checker type: " + type );
		}
	}
	
	return optimizer;
}

ModularOptimizer::Ptr parse_modular_optimizer( const ros::NodeHandle& ph )
{
	return ParseModularOptimizer( ph );
}

ModularOptimizer::Ptr parse_modular_optimizer( const YAML::Node& node )
{
	return ParseModularOptimizer( node );
}

}