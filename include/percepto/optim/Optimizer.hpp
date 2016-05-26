#pragma once

#include "percepto/compo/BackpropInfo.hpp"

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
		double value;
		VectorType gradient, step, params;
		do
		{
			value = _objective.Evaluate();
			VectorType gradient = BackpropGradient( _objective );
			step = _stepper.GetStep( -gradient );
			params = _objective.GetParamsVec() + step;
			_objective.SetParamsVec( params );
		}
		while( !_convergence.Converged( value, params, gradient ) );
	}

private:

	ObjectiveType& _objective; /**< A reference to this optimizer's object. */
	StepperType _stepper; /**< A reference to this optimizer's stepper. */
	ConvergenceType _convergence; /**< A reference to this optimizer's convergence object. */

};

}