#pragma once

#include "optim/OptimizationProfiler.h"
#include "optim/OptimInterfaces.h"
#include <deque>

namespace percepto
{

struct OptimizationResults
{
	std::string status; // Human-readable termination status
	double initialObjective;
	double finalObjective;
};

class ModularOptimizer
{
public:

	typedef std::shared_ptr<ModularOptimizer> Ptr;

	ModularOptimizer();

	OptimizationResults Optimize( OptimizationProblem& problem );

	// Resets all the optimizer modules
	void ResetAll();
	void ResetTerminationCheckers();
	void ResetSearchDirector();
	void ResetSearchStepper();

	// Set or add the various modules
	void SetSearchDirector( const SearchDirector::Ptr& director );
	void SetSearchStepper( const SearchStepper::Ptr& stepper );
	void AddTerminationChecker( const TerminationChecker::Ptr& checker );

private:

	OptimizationProfiler _profiler;

	SearchDirector::Ptr _searchDirector;
	SearchStepper::Ptr _searchStepper;

	// NOTE Use a deque to avoid memory re-allocation
	std::deque<TerminationChecker::Ptr> _terminationCheckers;
};

}