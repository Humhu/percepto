#pragma once

#include <iostream>

#include <Eigen/Core>
#include <nlopt.hpp>
#include <ctime>
#include "percepto/optim/OptimizationInterface.hpp"

namespace percepto
{

struct NLOptParameters
{
	nlopt::algorithm algorithm; // Name of algorithm to use
	double objStopValue; // Stop is func val drops below this
	double absFuncTolerance; // Stops if func change drops below this
	double relFuncTolerance; // Stops if func rel change less than this
	double absParamTolerance;
	double relParamTolerance; // Stops if rel parameter change less than this
	int maxFunctionEvals; // Stop if this many function evaluations
	double maxSeconds; // Stop after this many seconds

	NLOptParameters()
	: algorithm( nlopt::LD_LBFGS ),
	objStopValue( -std::numeric_limits<double>::infinity() ),
	absFuncTolerance( 0 ),
	relFuncTolerance( 1E-6 ),
	absParamTolerance( 0 ),
	relParamTolerance( 1E-6 ),
	maxFunctionEvals( -1 ),
	maxSeconds( std::numeric_limits<double>::infinity() ) {}
};

inline nlopt::algorithm StrToNLAlgo( const std::string& str )
{
	if( str == "GN_DIRECT") { return nlopt::GN_DIRECT; }
	else if( str == "GN_DIRECT_L") { return nlopt::GN_DIRECT_L; }
	else if( str == "GN_DIRECT_L_RAND") { return nlopt::GN_DIRECT_L_RAND; }
	else if( str == "GN_DIRECT_NOSCAL") { return nlopt::GN_DIRECT_NOSCAL; }
	else if( str == "GN_DIRECT_L_NOSCAL") { return nlopt::GN_DIRECT_L_NOSCAL; }
	else if( str == "GN_DIRECT_L_RAND_NOSCAL") { return nlopt::GN_DIRECT_L_RAND_NOSCAL; }
	else if( str == "GN_ORIG_DIRECT") { return nlopt::GN_ORIG_DIRECT; }
	else if( str == "GN_ORIG_DIRECT_L") { return nlopt::GN_ORIG_DIRECT_L; }
	else if( str == "GD_STOGO") { return nlopt::GD_STOGO; }
	else if( str == "GD_STOGO_RAND") { return nlopt::GD_STOGO_RAND; }
	else if( str == "LD_LBFGS_NOCEDAL") { return nlopt::LD_LBFGS_NOCEDAL; }
	else if( str == "LD_LBFGS") { return nlopt::LD_LBFGS; }
	else if( str == "LN_PRAXIS") { return nlopt::LN_PRAXIS; }
	else if( str == "LD_VAR1") { return nlopt::LD_VAR1; }
	else if( str == "LD_VAR2") { return nlopt::LD_VAR2; }
	else if( str == "LD_TNEWTON") { return nlopt::LD_TNEWTON; }
	else if( str == "LD_TNEWTON_RESTART") { return nlopt::LD_TNEWTON_RESTART; }
	else if( str == "LD_TNEWTON_PRECOND") { return nlopt::LD_TNEWTON_PRECOND; }
	else if( str == "LD_TNEWTON_PRECOND_RESTART") { return nlopt::LD_TNEWTON_PRECOND_RESTART; }
	else if( str == "GN_CRS2_LM") { return nlopt::GN_CRS2_LM; }
	else if( str == "GN_MLSL") { return nlopt::GN_MLSL; }
	else if( str == "GD_MLSL") { return nlopt::GD_MLSL; }
	else if( str == "GN_MLSL_LDS") { return nlopt::GN_MLSL_LDS; }
	else if( str == "GD_MLSL_LDS") { return nlopt::GD_MLSL_LDS; }
	else if( str == "LD_MMA") { return nlopt::LD_MMA; }
	else if( str == "LN_COBYLA") { return nlopt::LN_COBYLA; }
	else if( str == "LN_NEWUOA") { return nlopt::LN_NEWUOA; }
	else if( str == "LN_NEWUOA_BOUND") { return nlopt::LN_NEWUOA_BOUND; }
	else if( str == "LN_NELDERMEAD") { return nlopt::LN_NELDERMEAD; }
	else if( str == "LN_SBPLX") { return nlopt::LN_SBPLX; }
	else if( str == "LN_AUGLAG") { return nlopt::LN_AUGLAG; }
	else if( str == "LD_AUGLAG") { return nlopt::LD_AUGLAG; }
	else if( str == "LN_AUGLAG_EQ") { return nlopt::LN_AUGLAG_EQ; }
	else if( str == "LD_AUGLAG_EQ") { return nlopt::LD_AUGLAG_EQ; }
	else if( str == "LN_BOBYQA") { return nlopt::LN_BOBYQA; }
	else if( str == "GN_ISRES") { return nlopt::GN_ISRES; }
	else if( str == "AUGLAG") { return nlopt::AUGLAG; }
	else if( str == "AUGLAG_EQ") { return nlopt::AUGLAG_EQ; }
	else if( str == "G_MLSL") { return nlopt::G_MLSL; }
	else if( str == "G_MLSL_LDS") { return nlopt::G_MLSL_LDS; }
	else if( str == "LD_SLSQP") { return nlopt::LD_SLSQP; }
	else if( str == "LD_CCSAQ") { return nlopt::LD_CCSAQ; }
	else if( str == "GN_ESCH") { return nlopt::GN_ESCH; }
	else { throw std::runtime_error("Invalid algorithm string."); }
}

/*! \brief Optimization interface for NLOpt*/
template <typename CostType>
class NLOptInterface
{
public:

	typedef typename CostType::RegressorType RegressorType;
	//typedef typename RegressorType::ParamType RegressorParamType;
	//typedef OptimizationResults<RegressorParamType> ResultsType;
	typedef OptimizationResults ResultsType;

	// TODO Add progress printouts for verbose mode
	/*! \brief Create an optimizer interface with the specified algorithm. */
	NLOptInterface( CostType& cost, const NLOptParameters& params )
	: _cost( cost ), _verbose( false ), 
	_optimizer( params.algorithm, cost.ParamDim() )
	{
		_optimizer.set_min_objective( &NLOptInterface::ObjectiveCallback, this );
		_optimizer.set_stopval( params.objStopValue );
		_optimizer.set_ftol_abs( params.absFuncTolerance );
		_optimizer.set_ftol_rel( params.relFuncTolerance );
		_optimizer.set_xtol_abs( params.absParamTolerance );
		_optimizer.set_xtol_rel( params.relParamTolerance );
		_optimizer.set_maxeval( params.maxFunctionEvals );
		_optimizer.set_maxtime( params.maxSeconds );
	}

	void SetVerbosity( bool v ) { _verbose = v; }

	CostType& GetCost() { return _cost; }
	const CostType& GetCost() const { return _cost; }

	std::string GetAlgorithmName() 
	{ 
		return std::string( _optimizer.get_algorithm_name() ); 
	}

	/*! \brief Begins the optimization */
	ResultsType Optimize()
	{
		_profiler.StartOverall();
		
		double finalVal;

		//VectorType paramVec = reg.GetParamsVec();
		// for( unsigned int i = 0; i < paramVec.size(); i++ )
		// {
		// 	params[i] = paramVec(i);
		// }
		std::vector<double> params( _cost.ParamDim() );
		Eigen::Map<VectorType> paramsMap( params.data(), _cost.ParamDim(), 1 );
		paramsMap = _cost.GetParamsVec();
		
		_optimizer.optimize( params, finalVal );
		_profiler.StopOverall();

		ResultsType results = _profiler.GetResults();
		
		// for( unsigned int i = 0; i < paramVec.size(); i++ )
		// {
		// 	paramVec(i) = params[i];
		// }
		_cost.SetParamsVec( paramsMap );

		results.finalObjective = finalVal;
		return results;
	}

private:

	typedef OptimizationProfiler<ResultsType> ProfilerType;

	/*! \brief Objective and gradient function used by NLOpt. */
	static double ObjectiveCallback( unsigned n, const double* x, 
	                                 double* grad, void* f_data )
	{
		assert( f_data != nullptr );
		NLOptInterface* obj = static_cast<NLOptInterface*>( f_data );

		assert( n == obj->_cost.ParamDim() );

		Eigen::Map<const VectorType> paramVec( x, n, 1 );
		obj->_cost.SetParamsVec( paramVec );
		// std::cout << "params: " << paramVec.transpose() << std::endl;

		obj->_profiler.StartObjectiveCall();
		double objective = obj->_cost.Evaluate();
		// std::cout << "obj: " << objective << std::endl;
		obj->_profiler.FinishObjectiveCall();

		// NLOpt requests gradient by giving non-null pointer
		if( grad != nullptr )
		{
			// Efficient batch implementation
			obj->_profiler.StartGradientCall();
			//VectorType gradient = obj->_cost.Gradient();
			VectorType gradient = BackpropGradient( obj->_cost );
			for( unsigned int ind = 0; ind < n; ind++ )
			{
				grad[ind] = gradient(ind);
			}
			// std::cout << "grad: " << gradient.transpose() << std::endl;
			obj->_profiler.FinishGradientCall();
		}


		return objective;
	}

	ProfilerType _profiler;
	CostType& _cost;
	bool _verbose;
	nlopt::opt _optimizer;

};

}