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

	template <typename Problem>
	bool StepOnce( Problem& problem )
	{
		_profiler.StartObjectiveCall();
		problem.Invalidate();
		problem.Foreprop();
		double value = problem.GetOutput();
		_profiler.FinishObjectiveCall();

		_profiler.StartGradientCall();
		problem.Backprop();
		MatrixType dodw = _parametric.GetDerivs();
		_parametric.ResetAccumulators();
		if( dodw.rows() != 1 )
		{
			throw std::runtime_error( "Cost derivative dimension error." );
		}
		VectorType gradient = VectorType( dodw.transpose() );
		_profiler.FinishGradientCall();

		VectorType step = _stepper.GetStep( -gradient );
		VectorType params = _parametric.GetParamsVec();
		if( params.size() != step.size() )
		{
			throw std::runtime_error( "Parameter step the wrong size!" );
		}

		_parametric.SetParamsVec( params + step );

		bool converged = _convergence.Converged( value, params, step );
		bool failed = _convergence.Failed();

		return converged || failed;
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
		bool converged = false;
		bool failed = false;
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
			// if( !step.allFinite() )
			// {
			// 	std::cout << "cost: " << value << std::endl;
			// 	std::cout << "params: " << params.transpose() << std::endl;
			// 	std::cout << step.transpose() << std::endl;
			// 	throw std::runtime_error( "Non-finite step!" );
			// }

			_parametric.SetParamsVec( params + step );

			converged = _convergence.Converged( value, params, step );
			failed = _convergence.Failed();
		}
		while( !converged && !failed );

		_profiler.StopOverall();
		OptimizationResults results = _profiler.GetResults();
		results.finalObjective = value;
		results.converged = converged;
		return results;
	}

private:

	OptimizationProfiler _profiler;
	StepperType& _stepper; /**< A reference to this optimizer's stepper. */
	ConvergenceType& _convergence; /**< A reference to this optimizer's convergence object. */
	Parameters& _parametric;

};

}