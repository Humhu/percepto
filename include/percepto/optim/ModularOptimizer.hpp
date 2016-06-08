#pragma once

#include "percepto/compo/Parametric.hpp"
#include "percepto/optim/OptimizationInterface.hpp"

#include <iostream>
#include <functional>

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
	ModularOptimizer( StepperType& stepper, ConvergenceType& convergence,
	                  Parametric& parametrics )
	: _stepper( stepper ), _convergence( convergence ), _parametric( parametrics )
	{
		Initialize();
	}

	void Initialize()
	{
		_profiler.StartOverall();
		_stepper.Reset();
		_convergence.Reset();
	}

	template <typename CostType>
	bool StepOnce( CostType& cost )
	{
		
		_profiler.StartObjectiveCall();
		double value = cost.Evaluate();
		_profiler.FinishObjectiveCall();

		_profiler.StartGradientCall();
		MatrixType sysDodx = MatrixType::Identity(1,1);
		cost.Backprop( sysDodx );
		MatrixType dodw = _parametric.GetAccWeightDerivs();
		if( dodw.rows() != 1 )
		{
			throw std::runtime_error( "Cost derivative dimension error." );
		}
		VectorType gradient( dodw.transpose() );

		_parametric.ResetAccumulators();
		_profiler.FinishGradientCall();

		VectorType step = _stepper.GetStep( -gradient );
		VectorType params = _parametric.GetParamsVec();
		if( params.size() != step.size() )
		{
			throw std::runtime_error( "Parameter step the wrong size!" );
		}
		_parametric.SetParamsVec( params + step );

		return _convergence.Converged( value, params, gradient );
	}

	OptimizationResults GetResults()
	{
		_profiler.StopOverall();
		return _profiler.GetResults();
	}

	/** 
	 * @brief Begin optimization by stepping the cost's parameters until
	 * convergence is reached. */
	template <typename CostType>
	OptimizationResults Optimize( CostType& cost )
	{
		_convergence.Reset();
		_profiler.StartOverall();

		double value;
		VectorType gradient, step, params;
		MatrixType sysDodx = MatrixType::Identity(1,1);
		do
		{
			_profiler.StartObjectiveCall();
			value = cost.Evaluate();
			_profiler.FinishObjectiveCall();

			_profiler.StartGradientCall();
			//gradient = BackpropGradient( cost );
			cost.Backprop( sysDodx );
			MatrixType dodw = _parametric.GetAccWeightDerivs();
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
	Parametric& _parametric;

};

}