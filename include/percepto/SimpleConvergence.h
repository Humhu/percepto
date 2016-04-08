#pragma once

#include "percepto/PerceptoTypes.hpp"
#include <ctime>

namespace percepto
{

struct SimpleConvergenceCriteria
{
	// Max runtime (Inf)
	double maxRuntime;
	// Max number of function/gradient evaluations (0)
	unsigned int maxIterations;
	// Biggest single element delta must be less than (-Inf)
	double minElementDelta; 
	// Average parameter delta must be less than (-Inf)
	double minAverageDelta;
	// Biggest single gradient element must be less than (-Inf)
	double minElementGradient;
	// Average gradient must be less than (-Inf)
	double minAverageGradient;
	// Objective change must be less than (-Inf)
	double minObjectiveDelta;

	SimpleConvergenceCriteria();
};

/*! \brief Represents basic convergence criterion for optimization. */
class SimpleConvergence
{
public:

	SimpleConvergence( const SimpleConvergenceCriteria& critiera );

	bool Converged( double objective, const VectorType& params,
	                const VectorType& gradient );

private:

	SimpleConvergenceCriteria _criteria;

	bool _initialized;
	unsigned int _iteration;
	double _lastObjective;
	VectorType _lastGradient;
	VectorType _lastParams;
	clock_t _startTicks;
};

}