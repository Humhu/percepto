#pragma once

#include "percepto/compo/Parametric.hpp"
#include "percepto/optim/OptimizationInterface.hpp"

#include <iostream>
#include <functional>

namespace percepto
{

/** 
 * @brief Basic modular optimization framework class. Minimizes an problem
 * with a gradient stepping policy while checking convergence. 
 * @tparam Cost The problem class.
 * @tparam Stepper The stepping policy class.
 * @tparam Convergence The problem convergence class. 
 */
template <typename Stepper, typename Convergence>
class ModularOptimizer
{
public:

	typedef Stepper StepperType;
	typedef Convergence ConvergenceType;

	/** 
	 * @brief Create an optimization object with references to an problem,
	 * stepper, and convergence object. Keeps references only. */
	ModularOptimizer( StepperType& stepper, ConvergenceType& convergence,
	                  Parameters& parameters )
	: _stepper( stepper ), _convergence( convergence ), _parametric( parameters )
	{
		Initialize();
	}

	void Initialize()
	{
		_profiler.StartOverall();
		_stepper.Reset();
		_convergence.Reset();
	}

	OptimizationResults GetResults()
	{
		_profiler.StopOverall();
		return _profiler.GetResults();
	}

	/** 
	 * @brief Begin optimization by stepping the problem's parameters until
	 * convergence is reached. */
	template <typename Problem>
	OptimizationResults Optimize( Problem& problem )
	{
		_convergence.Reset();
		_profiler.StartOverall();

		double value;
		VectorType gradient, step, params;
		MatrixType sysDodx = MatrixType::Identity(1,1);
		do
		{
			_profiler.StartObjectiveCall();
			problem.Invalidate();
			problem.Foreprop();
			value = problem.GetOutput();
			_profiler.FinishObjectiveCall();

			_profiler.StartGradientCall();
			problem.Backprop();
			MatrixType dodw = _parametric.GetDerivs();
			_parametric.ResetAccumulators();
			if( dodw.rows() != 1 )
			{
				throw std::runtime_error( "Cost derivative dimension error." );
			}
			gradient = VectorType( dodw.transpose() );
			_profiler.FinishGradientCall();

			step = _stepper.GetStep( -gradient );
			params = _parametric.GetParamsVec();
			if( params.size() != step.size() )
			{
				throw std::runtime_error( "Parameter step the wrong size!" );
			}
			_parametric.SetParamsVec( params + step );
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
	Parameters& _parametric;

};

}