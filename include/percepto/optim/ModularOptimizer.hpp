#pragma once

#include "percepto/compo/BackpropInfo.hpp"
#include "percepto/optim/OptimizationInterface.hpp"

namespace percepto
{

/** 
 * @brief Basic modular optimization framework class. Minimizes an cost
 * with a gradient stepping policy while checking convergence. 
 * @tparam Cost The cost class.
 * @tparam Stepper The stepping policy class.
 * @tparam Convergence The cost convergence class. 
 */
template <typename Stepper, typename Convergence>
class ModularOptimizer
{
public:

	typedef Stepper StepperType;
	typedef Convergence ConvergenceType;

	/** 
	 * @brief Create an optimization object with references to an cost,
	 * stepper, and convergence object. Keeps references only. */
	ModularOptimizer( StepperType& stepper, ConvergenceType& convergence )
	: _stepper( stepper ), _convergence( convergence ) {}

	/** 
	 * @brief Begin optimization by stepping the cost's parameters until
	 * convergence is reached. */
	template <typename CostType>
	OptimizationResults Optimize( CostType& cost )
	{
		_profiler.StartOverall();

		double value;
		VectorType gradient, step, params;
		do
		{
			_profiler.StartObjectiveCall();
			value = cost.Evaluate();
			_profiler.FinishObjectiveCall();

			_profiler.StartGradientCall();
			VectorType gradient = BackpropGradient( cost );
			_profiler.FinishGradientCall();

			step = _stepper.GetStep( -gradient );
			params = cost.GetParamsVec() + step;
			cost.SetParamsVec( params );
		}
		while( !_convergence.Converged( value, params, gradient ) );

		_profiler.StopOverall();
		OptimizationResults results = _profiler.GetResults();
		results.finalObjective = value;
		return results;
	}

private:

	OptimizationProfiler _profiler;
	StepperType& _stepper; /**< A reference to this optimizer's stepper. */
	ConvergenceType& _convergence; /**< A reference to this optimizer's convergence object. */

};

}