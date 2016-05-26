#pragma once

#include "percepto/PerceptoTypes.h"
#include "percepto/Backprop.hpp"

namespace percepto
{

/** 
 * @brief Basic optimization framework class. Minimizes an objective
 * with a gradient stepping policy while checking convergence. 
 * @tparam Objective The objective class.
 * @tparam Stepper The stepping policy class.
 * @tparam Convergence The objective convergence class. 
 */
template <typename Objective, typename Stepper, typename Convergence>
class Optimizer
{
public:

	typedef Objective ObjectiveType;
	typedef Stepper StepperType;
	typedef Convergence ConvergenceType;

	/** 
	 * @brief Create an optimization object with references to an objective,
	 * stepper, and convergence object. Keeps references only. */
	Optimizer( ObjectiveType& objective, const StepperType& stepper, 
	           const ConvergenceType& convergence )
	: _objective( objective ), _stepper( stepper ), _convergence( convergence ) {}

	/** 
	 * @brief Begin optimization by stepping the objective's parameters until
	 * convergence is reached. */
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
		}
		while( !_convergence.Converged( objective, params, gradient ) );
	}

private:

	ObjectiveType& _objective; /**< A reference to this optimizer's object. */
	StepperType _stepper; /**< A reference to this optimizer's stepper. */
	ConvergenceType _convergence; /**< A reference to this optimizer's convergence object. */

};

}