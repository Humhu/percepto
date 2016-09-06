#pragma once

#include <lbfgs.h>
#include <stdexcept>
#include "modprop/optim/OptimizationInterface.hpp"

namespace percepto
{

// TODO Update
/*! \brief Optimization interface for Lib LBFGS */
template <typename CostType>
class LlbfgsInterface
{
public:

	typedef typename CostType::ScalarType ScalarType;
	typedef OptimizationResults<CostType> ResultsType;
	typedef typename CostType::ParameterType ParameterType;

	/*! \brief Create an optimizer interface with the specified algorithm. */
	LlbfgsInterface( CostType& r )
	: _cost( r ), _verbose( false )
	{
		// TODO Make configurable
		// Initialize lbfgs
		lbfgs_parameter_init( &_lbfgs );
	}

	void SetVerbosity( bool v ) { _verbose = v; }

	std::string GetAlgorithmName() 
	{ 
		return "L-BFGS (liblbfgs)"; 
	}

	ResultsType Optimize( const ParameterType& initParams )
	{
		_profiler.StartOverall();

		lbfgsfloatval_t* x = AllocateVars( initParams.size() );
		EigenToFloat( initParams, x );
		lbfgsfloatval_t* fx = AllocateVars( 1 );

		// TODO Parse retval to return status
		int retval = lbfgs( initParams.size(),
		                    x,
		                    fx,
		                    &LlbfgsInterface::EvaluateCallback,
		                    &LlbfgsInterface::ProgressCallback,
		                    this,
		                    &_lbfgs );

		_profiler.StopOverall();

		ResultsType results = _profiler.GetResults();
		results.initialParameters = initParams;
		results.finalParameters = ParameterType( initParams.size() );
		FloatToEigen( x, results.finalParameters );
		results.finalObjective = -*fx; // Must negate since llbfgs minimizes
		return results;
	}

private:

	typedef OptimizationProfiler<CostType> ProfilerType;

	// Copy in x to take care of casting. Alternatively can use a map and
	// matrix cast, but this is more clear.
	template <class Derived>
	static void EigenToFloat( const Eigen::DenseBase<Derived>& src,
	                   lbfgsfloatval_t* dst )
	{
		for( unsigned int i = 0; i < src.size(); i++ )
		{
			dst[i] = src(i);
		}
	}

	template <class Derived>
	static void FloatToEigen( const lbfgsfloatval_t* src,
	                   Eigen::DenseBase<Derived>& dst )
	{
		for( unsigned int i = 0; i < dst.size(); i++ )
		{
			dst(i) = src[i];
		}
	}

	lbfgsfloatval_t* AllocateVars( int n )
	{
		lbfgsfloatval_t* var = lbfgs_malloc( n );
		if( !var ) 
		{
			throw std::bad_alloc();
		}
		return var;
	}

	/*! \brief Evaluation callback used by lbfgs during optimization. */
	static lbfgsfloatval_t EvaluateCallback( void *instance, 
	                                         const lbfgsfloatval_t *x, 
	                                         lbfgsfloatval_t *g, 
	                                         const int n, 
	                                         const lbfgsfloatval_t step )
	{
		assert( instance != nullptr );
		LlbfgsInterface* obj = static_cast<LlbfgsInterface*>( instance );
		
		assert( n == obj->_cost.ParameterDim() );
		ParameterType currentParams( n );
		FloatToEigen( x, currentParams );

		obj->_cost.SetParameters( currentParams );

		// NOTE lbfgs wants to minimize - we want to maximize
		obj->_profiler.StartObjectiveCall();
		lbfgsfloatval_t retval = -obj->_cost.Evaluate();
		obj->_profiler.FinishObjectiveCall();

		// NOTE lbfgs wants to minimize - we want to maximize
		obj->_profiler.StartGradientCall();
		ParameterType gradient = -obj->_cost.Gradient();
		EigenToFloat( gradient, g );
		obj->_profiler.FinishGradientCall();

		return retval;
	}

	/*! \brief Progress callback used by lbfgs during optimization. Returns 0
	 * to continue optimizing, non-zero to stop. */
	static int ProgressCallback( void *instance, 
	                             const lbfgsfloatval_t *x, 
	                             const lbfgsfloatval_t *g, 
	                             const lbfgsfloatval_t fx, 
	                             const lbfgsfloatval_t xnorm, 
	                             const lbfgsfloatval_t gnorm, 
	                             const lbfgsfloatval_t step, 
	                             int n, int k, int ls )
	{
		assert( instance != nullptr );
		LlbfgsInterface* obj = static_cast<LlbfgsInterface*>( instance );
		obj->_numEvals++;

		if( obj->_verbose )
		{
			std::cout << "Iteration: " << k << std::endl;
			std::cout << "\tgnorm: " << gnorm << std::endl;
		}

		// For now, just return when 
		// TODO Convergence criteria
		if( gnorm / n < 1E-6 ) { return 1; }
		return 0;
	}

	ProfilerType _profiler;
	CostType& _cost;
	bool _verbose;

	lbfgs_parameter_t _lbfgs;
	unsigned int _numEvals;

};

}
