#pragma once

#include "percepto/PerceptoTypes.hpp"
#include "percepto/Backprop.hpp"

namespace percepto
{

// Objective is the minimization objective
// Stepper is the stepping policy
// Convergence is the convergence policy
template <typename Objective, typename Stepper, typename Convergence>
class Optimizer
{
public:

	typedef Objective ObjectiveType;
	typedef Stepper StepperType;
	typedef Convergence ConvergenceType;

	// Keeps the objective by reference only
	Optimizer( ObjectiveType& objective, const StepperType& stepper, 
	           const ConvergenceType& convergence )
	: _objective( objective ), _stepper( stepper ), _convergence( convergence ) {}

	void Run()
	{
		double objective;
		VectorType gradient, step, params;
		do
		{
			_objective.EvaluateAndGradient( objective, gradient );
			step = _stepper.GetStep( -gradient );
			_objective.GetRegressor().StepParams( step );
			params = _objective.GetRegressor().GetParamsVec();

			std::cout << "objective: " << objective << std::endl;
		}
		while( !_convergence.Converged( objective, params, gradient ) );
	}

private:

	ObjectiveType& _objective;
	StepperType _stepper;
	ConvergenceType _convergence;

};

}