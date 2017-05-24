#include "optim/ModularOptimizer.h"
#include <boost/foreach.hpp>
#include <sstream>
#include <iostream>

namespace argus
{

ModularOptimizer::ModularOptimizer() {}

OptimizationResults ModularOptimizer::Optimize( OptimizationProblem& problem )
{
	_profiler.Reset();

	OptimizationResults results;
	results.status = "";

	// TODO Many problems need this initial resample to start up correctly
	// Move this into some sort of Initialize method instead?
	problem.Resample();
	results.initialObjective = problem.ComputeObjective();

	// An empty status denotes that we haven't terminated yet
	while( results.status.empty() )
	{
		problem.Resample();

		// std::cout << "Objective: " << problem.ComputeObjective() << std::endl;

		VectorType direction = _searchDirector->ComputeSearchDirection( problem );
		VectorType params = problem.GetParameters();
		if( !params.allFinite() )
		{
			std::stringstream ss;
			ss << "Parameters not finite: " << params.transpose();
			throw std::runtime_error( ss.str() );
		}

		if( direction.size() != params.size() )
		{
			throw std::runtime_error( "Incorrect search direction dimensionality.");
		}

		if( !direction.allFinite() )
		{
			std::stringstream ss;
			ss << "Direction not finite: " << direction.transpose();
			throw std::runtime_error( ss.str() );
		}

		double stepSize = _searchStepper->ComputeStepSize( problem, direction );

		if( !std::isfinite( stepSize ) )
		{
			throw std::runtime_error( "Non-finite step size returned." );
		}

		problem.SetParameters( params + stepSize * direction );

		BOOST_FOREACH( const TerminationChecker::Ptr& checker, _terminationCheckers )
		{
			std::string status = checker->CheckTermination( problem );
			if( !status.empty() )
			{
				results.status = status;
				break;
			}
		}
	}

	results.finalObjective = problem.ComputeObjective();
	return results;
}

void ModularOptimizer::ResetAll()
{
	ResetTerminationCheckers();
	ResetSearchDirector();
	ResetSearchStepper();
}

void ModularOptimizer::ResetTerminationCheckers()
{
	BOOST_FOREACH( const TerminationChecker::Ptr& checker, _terminationCheckers )
	{
		checker->Reset();
	}
}

void ModularOptimizer::ResetSearchDirector()
{
	_searchDirector->Reset();
}

void ModularOptimizer::ResetSearchStepper()
{
	_searchStepper->Reset();
}

void ModularOptimizer::SetSearchDirector( const SearchDirector::Ptr& director )
{
	_searchDirector = director;
}

void ModularOptimizer::SetSearchStepper( const SearchStepper::Ptr& stepper )
{
	_searchStepper = stepper;
}

void ModularOptimizer::AddTerminationChecker( const TerminationChecker::Ptr& checker )
{
	_terminationCheckers.push_back( checker );
}

}