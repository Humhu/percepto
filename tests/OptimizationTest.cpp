#include "percepto/compo/LinearRegressor.hpp"
#include "percepto/compo/ExponentialWrapper.hpp"
#include "percepto/compo/AffineWrapper.hpp"

#include "percepto/utils/LowerTriangular.hpp"
#include "percepto/compo/ModifiedCholeskyWrapper.hpp"

#include "percepto/optim/GaussianLogLikelihoodCost.hpp"
#include "percepto/optim/MeanPopulationCost.hpp"
#include "percepto/optim/ParameterL2Cost.hpp"
#include "percepto/optim/NlOptInterface.hpp"
#include "percepto/optim/OptimizerTypes.h"

#include "percepto/utils/MultivariateGaussian.hpp"
#include "percepto/utils/Randomization.hpp"
#include "percepto/neural/NetworkTypes.h"

#include <ctime>
#include <iostream>

using namespace percepto;

typedef ReLUNet BaseRegressor;
typedef ExponentialWrapper<BaseRegressor> ExpReg;
typedef ModifiedCholeskyWrapper<BaseRegressor, ExpReg> ModCholReg;

typedef AffineWrapper<ModCholReg> AffineModCholReg;

typedef GaussianLogLikelihoodCost<AffineModCholReg> GLL;
typedef MeanPopulationCost<GLL> MeanGLL;
typedef ParameterL2Cost<MeanGLL> MeanGLL_L2;

typedef NLOptInterface NLOptimizer;

template <class Optimizer, typename Cost>
void TestOptimization( Optimizer& opt, Cost& cost,
                       const VectorType& initParams,
                       const VectorType& trueParams )
{
	std::cout << "Beginning test of with " << cost.ParamDim() << " parameters... " << std::endl;

	// opt.SetVerbosity( false );
	OptimizationResults results;
	try
	{
		cost.SetParamsVec( initParams );
		results = opt.Optimize( cost );
	}
	catch( std::runtime_error e )
	{
		std::cout << "Received exception: " << e.what() << std::endl;
		return;
	}

	VectorType finalParams = cost.GetParamsVec();
	VectorType delta = trueParams - finalParams;
	double errorNorm = delta.norm() / delta.size();
	double errorMax = std::max( -delta.minCoeff(), delta.maxCoeff() );

	cost.SetParamsVec( trueParams );
	double minCost = cost.Evaluate();

	std::cout << "True params: " << std::endl << trueParams.transpose() << std::endl;
	std::cout << "Final params: " << std::endl << finalParams.transpose() << std::endl;
	std::cout << "\tFinal objective: " << results.finalObjective << std::endl;
	std::cout << "\tTrue objective: " << minCost << std::endl;
	std::cout << "\tOverall time: " << results.totalElapsedSecs << std::endl;
	std::cout << "\tObjective time: " << results.totalObjectiveSecs << std::endl;
	std::cout << "\tGradient time: " << results.totalGradientSecs << std::endl;
	std::cout << "\tObjective evaluations: " << results.numObjectiveEvaluations << std::endl;
	std::cout << "\tGradient evaluations: " << results.numGradientEvaluations << std::endl;
	std::cout << "\tAvg time/objective: " << results.totalObjectiveSecs/results.numObjectiveEvaluations << std::endl;
	std::cout << "\tAvg time/gradient: " << results.totalGradientSecs/results.numGradientEvaluations << std::endl;
	std::cout << "\tAverage error norm: " << errorNorm << std::endl;
	std::cout << "\tMax error: " << errorMax << std::endl;
}

int main( void )
{

	unsigned int matDim = 6;
	unsigned int lFeatDim = 5;
	unsigned int dFeatDim = 5;
	unsigned int lOutDim = TriangularMapping::num_positions( matDim - 1 );
	unsigned int dOutDim = matDim;

	unsigned int lNumHiddenLayers = 1;
	unsigned int lLayerWidth = 10;
	unsigned int dNumHiddenLayers = 1;
	unsigned int dLayerWidth = 10;

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "L feature dim: " << lFeatDim << std::endl;
	std::cout << "D feature dim: " << dFeatDim << std::endl;
	std::cout << "L output dim: " << lOutDim << std::endl;
	std::cout << "D output dim: " << dOutDim << std::endl;

	MatrixType mcOffset = 1E-2 * MatrixType::Identity( matDim, matDim );

	// True model
	HingeActivation relu( 1.0, 1E-3 );
	BaseRegressor trueLreg = BaseRegressor::create_zeros( lFeatDim, lOutDim, 
	                                                      lNumHiddenLayers, lLayerWidth, relu );
	VectorType params = trueLreg.GetParamsVec();
	randomize_vector( params );
	trueLreg.SetParamsVec( params );

	BaseRegressor trueDreg = BaseRegressor::create_zeros( dFeatDim, dOutDim, 
	                                                      dNumHiddenLayers, dLayerWidth, relu );
	params = trueDreg.GetParamsVec();
	randomize_vector( params );
	trueDreg.SetParamsVec( params );

	ExpReg trueExpreg( trueDreg );
	ModCholReg trueMc( trueLreg, trueExpreg, mcOffset );
	AffineModCholReg trueModel( trueMc );
	VectorType trueParams = trueMc.GetParamsVec();

	// Test weight creation
	VectorType weights = trueMc.CreateWeightVector( 0.1, 0.2 );
	std::cout << "weights: " << weights.transpose() << std::endl;

	// Initial model
	BaseRegressor lreg = BaseRegressor::create_zeros( lFeatDim, lOutDim, 
	                                                      lNumHiddenLayers, lLayerWidth, relu );
	params = lreg.GetParamsVec();
	randomize_vector( params );
	lreg.SetParamsVec( params );

	BaseRegressor dreg = BaseRegressor::create_zeros( dFeatDim, dOutDim, 
	                                                      dNumHiddenLayers, dLayerWidth, relu );
	params = dreg.GetParamsVec();
	randomize_vector( params );
	dreg.SetParamsVec( params );

	ExpReg expreg( dreg );
	ModCholReg initMc( lreg, expreg, mcOffset );
	AffineModCholReg initModel( initMc );

	// Create test population
	unsigned int popSize = 1000;
	std::cout << "Sampling " << popSize << " datapoints..." << std::endl;
	std::vector<GLL> baseCosts;
	baseCosts.reserve( popSize );
	MultivariateGaussian<> mvg( MultivariateGaussian<>::VectorType::Zero( matDim ),
	                            MultivariateGaussian<>::MatrixType::Identity( matDim, matDim ) );

	for( unsigned int i = 0; i < popSize; i++ )
	{
		typedef ModCholReg::LRegressorType::InputType LInputType;
		typedef ModCholReg::DRegressorType::InputType DInputType;
		
		ModCholReg::InputType mcInput;
		mcInput.lInput = LInputType::Random( lFeatDim );
		mcInput.dInput = DInputType::Random( dFeatDim );

		AffineModCholReg::InputType amcInput;
		amcInput.baseInput = mcInput;
		amcInput.transform = MatrixType::Identity( matDim, matDim );
		amcInput.offset = 1E-3 * MatrixType::Identity( matDim, matDim );

		mvg.SetCovariance( trueModel.Evaluate( amcInput ) );
		VectorType sample = mvg.Sample(); 
		baseCosts.emplace_back( amcInput, sample, initModel );
	}
	std::cout << "Sampling complete." << std::endl;

	MeanGLL meanCost( baseCosts );
	MeanGLL_L2 finalCost( meanCost, 1E-3 );

	VectorType initParams = initModel.GetParamsVec();

	NLOptParameters optParams;

	// Benchmark the different methods
	std::vector<nlopt::algorithm> algorithms =
	{ 
	  //nlopt::LD_SLSQP, 
	  nlopt::LD_LBFGS, 
	  //nlopt::LD_VAR2 
	};
	BOOST_FOREACH( nlopt::algorithm algo, algorithms )
	{
		optParams.algorithm = algo;
		NLOptimizer opt( optParams );
		TestOptimization( opt, finalCost, initParams, trueParams );
	}

	AdamStepper stepper;
	
	SimpleConvergenceCriteria criteria;
	criteria.maxRuntime = 60;
	criteria.minElementGradient = 1E-3;
	criteria.minObjectiveDelta = 1E-3;
	SimpleConvergence convergence( criteria );

	AdamOptimizer opt( stepper, convergence );
	TestOptimization( opt, finalCost, initParams, trueParams );

}