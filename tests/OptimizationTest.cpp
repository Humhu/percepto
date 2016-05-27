#include "percepto/compo/ConstantRegressor.hpp"
#include "percepto/compo/ExponentialWrapper.hpp"
#include "percepto/compo/ModifiedCholeskyWrapper.hpp"
#include "percepto/compo/TransformWrapper.hpp"
#include "percepto/compo/OffsetWrapper.hpp"
#include "percepto/compo/InputWrapper.hpp"

#include "percepto/utils/LowerTriangular.hpp"

#include "percepto/optim/ParameterL2Cost.hpp"
#include "percepto/optim/GaussianLogLikelihoodCost.hpp"
#include "percepto/optim/MeanPopulationCost.hpp"
#include "percepto/optim/StochasticPopulationCost.hpp"

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
typedef ModifiedCholeskyWrapper<ConstantRegressor, ExpReg> PSDReg;
typedef OffsetWrapper<PSDReg> PDReg;

typedef TransformWrapper<PDReg> TransPDReg;
// TODO Test with offsets also
typedef InputWrapper<TransPDReg> CovEst;
typedef GaussianLogLikelihoodCost<CovEst> GLL;

typedef MeanPopulationCost<GLL> MeanGLL;
typedef StochasticPopulationCost<GLL> StochasticGLL;
typedef ParameterL2Cost<MeanGLL> PenalizedMeanGLL;
typedef ParameterL2Cost<StochasticGLL> PenalizedStochasticGLL;

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
	unsigned int lFeatDim = 1;
	unsigned int dFeatDim = 5;
	unsigned int lOutDim = TriangularMapping::num_positions( matDim - 1 );
	unsigned int dOutDim = matDim;

	unsigned int dNumHiddenLayers = 1;
	unsigned int dLayerWidth = 10;

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "L feature dim: " << lFeatDim << std::endl;
	std::cout << "D feature dim: " << dFeatDim << std::endl;
	std::cout << "L output dim: " << lOutDim << std::endl;
	std::cout << "D output dim: " << dOutDim << std::endl;

	MatrixType pdOffset = 1E-2 * MatrixType::Identity( matDim, matDim );

	// True model
	ConstantRegressor trueLreg = ConstantRegressor( MatrixType( lOutDim, 1 ) );
	VectorType params = trueLreg.GetParamsVec();
	randomize_vector( params );
	trueLreg.SetParamsVec( params );

	HingeActivation relu( 1.0, 1E-3 );
	BaseRegressor trueDreg = BaseRegressor::create_zeros( dFeatDim, dOutDim, 
	                                                      dNumHiddenLayers, 
	                                                      dLayerWidth, relu );
	params = trueDreg.GetParamsVec();
	randomize_vector( params );
	trueDreg.SetParamsVec( params );

	ExpReg trueExpReg( trueDreg );
	PSDReg truePsdReg( trueLreg, trueExpReg );
	PDReg truePdReg( truePsdReg, pdOffset );
	VectorType trueParams = truePdReg.GetParamsVec();

	// Initial model
	ConstantRegressor lreg( MatrixType::Zero( lOutDim, 1 ) );

	BaseRegressor dreg = BaseRegressor::create_zeros( dFeatDim, dOutDim, 
	                                                  dNumHiddenLayers, 
	                                                  dLayerWidth, relu );
	params = dreg.GetParamsVec();
	randomize_vector( params );
	dreg.SetParamsVec( params );

	ExpReg expReg( dreg );
	PSDReg psdReg( lreg, dreg );
	PDReg pdReg( psdReg, pdOffset );

	// Create test population
	unsigned int popSize = 1000;
	std::cout << "Sampling " << popSize << " datapoints..." << std::endl;
	
	std::vector<TransPDReg> transformWrappers;
	std::vector<CovEst> estimates;
	std::vector<GLL> baseCosts;
	transformWrappers.reserve( popSize );
	estimates.reserve( popSize );
	baseCosts.reserve( popSize );
	MultivariateGaussian<> mvg( MultivariateGaussian<>::VectorType::Zero( matDim ),
	                            MultivariateGaussian<>::MatrixType::Identity( matDim, matDim ) );

	for( unsigned int i = 0; i < popSize; i++ )
	{
		PDReg::InputType pdInput;
		pdInput.lInput = VectorType( lFeatDim );
		pdInput.dInput = VectorType( dFeatDim );
		randomize_vector( pdInput.lInput );
		randomize_vector( pdInput.dInput );

		MatrixType transform = MatrixType::Identity( matDim, matDim );
		MatrixType trueCov = transform * truePdReg.Evaluate( pdInput ) * transform.transpose();
		mvg.SetCovariance( trueCov );
		VectorType sample = mvg.Sample(); 

		transformWrappers.emplace_back( pdReg, transform );
		estimates.emplace_back( transformWrappers.back(), pdInput );
		baseCosts.emplace_back( estimates.back(), sample );
	}
	std::cout << "Sampling complete." << std::endl;

	MeanGLL meanCost( baseCosts );
	PenalizedMeanGLL penalizedMeanCosts( meanCost, 0 );

	unsigned int minibatchSize = 20;
	StochasticGLL stochasticCost( baseCosts, minibatchSize );
	PenalizedStochasticGLL penalizedStochasticCosts( stochasticCost, 1E-3 );

	VectorType initParams = pdReg.GetParamsVec();

	NLOptParameters optParams;
	optParams.algorithm = nlopt::LD_LBFGS;
	NLOptimizer nlOpt( optParams );
	TestOptimization( nlOpt, penalizedMeanCosts, initParams, trueParams );

	AdamStepper stepper;	
	SimpleConvergenceCriteria criteria;
	criteria.maxRuntime = 10;
	criteria.minElementGradient = 1E-3;
	criteria.minObjectiveDelta = 1E-3;
	SimpleConvergence convergence( criteria );

	AdamOptimizer modOpt( stepper, convergence );
	TestOptimization( modOpt, penalizedStochasticCosts, initParams, trueParams );

}