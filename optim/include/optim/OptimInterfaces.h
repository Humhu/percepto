#pragma once

#include "modprop/ModpropTypes.h"
#include <memory>

namespace argus
{

// Interface for all optimization problems
// TODO Handle stochastic problems
class OptimizationProblem
{
public:

	typedef std::shared_ptr<OptimizationProblem> Ptr;

	OptimizationProblem() {}
	virtual ~OptimizationProblem() {}

	// Returns whether this problem is a minimization problem
	// If false, it is taken to be a maximization problem
	virtual bool IsMinimization() const = 0;

	// Resample losses if problem is stochastic
	virtual void Resample() {};

	// Compute the current objective
	virtual double ComputeObjective() = 0;

	// Compute the current gradient
	virtual VectorType ComputeGradient() = 0;

	virtual VectorType GetParameters() const = 0;
	virtual void SetParameters( const VectorType& params ) = 0;
};

// Interface for all classes that return search directions
class SearchDirector
{
public:

	typedef std::shared_ptr<SearchDirector> Ptr;

	SearchDirector() {}
	virtual ~SearchDirector() {}

	// Reset any state in the search director
	virtual void Reset() = 0;

	// Returns the latest search direction, accounting for problem min/max direction
	virtual VectorType ComputeSearchDirection( OptimizationProblem& problem ) = 0;
};

// Interface for all classes that choose step sizes
class SearchStepper
{
public:

	typedef std::shared_ptr<SearchStepper> Ptr;

	SearchStepper() {}
	virtual ~SearchStepper() {}

	virtual void Reset() = 0;

	// Compute the appropriate scalar step size for the given direction
	virtual double ComputeStepSize( OptimizationProblem& problem,
	                                const VectorType& direction ) = 0;
};

// Interface for all classes that check for optimization termination
class TerminationChecker
{
public:

	typedef std::shared_ptr<TerminationChecker> Ptr;

	TerminationChecker() {}
	virtual ~TerminationChecker() {}

	// Check to see if the problem has converged
	// Returns an empty string if not terminated, otherwise termination status
	virtual std::string CheckTermination( OptimizationProblem& problem ) = 0;

	// Reset any termination checking criteria
	virtual void Reset() = 0;
};

}