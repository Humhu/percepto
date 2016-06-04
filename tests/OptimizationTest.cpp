#include "percepto/compo/ConstantRegressor.hpp"
#include "percepto/compo/ExponentialWrapper.hpp"
#include "percepto/compo/ModifiedCholeskyWrapper.hpp"
#include "percepto/compo/TransformWrapper.hpp"
#include "percepto/compo/AdditiveWrapper.hpp"
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
typedef RegressorOffsetWrapper<PSDReg> PDReg;

typedef InputWrapper<PDReg> CovEst;
typedef TransformWrapper<CovEst> TransCovEst;
typedef AdditiveWrapper<CovEst,CovEst> SumCovEst;
typedef GaussianLogLikelihoodCost<SumCovEst> GLL;

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
	
	double initialObjective = cost.Evaluate();

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

	cost.SetParamsVec( finalParams );

	std::cout << "True params: " << std::endl << trueParams.transpose() << std::endl;
	std::cout << "Final params: " << std::endl << finalParams.transpose() << std::endl;
	std::cout << "\tInitial objective: " << initialObjective << std::endl;
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
	HingeActivation relu( 1.0, 1E-3 );

	// True model
	// A
	ConstantRegressor trueLregA = ConstantRegressor( MatrixType( lOutDim, 1 ) );
	VectorType params = trueLregA.GetParamsVec();
	randomize_vector( params );
	trueLregA.SetParamsVec( params );

	BaseRegressor trueDregA( dFeatDim, dOutDim, dNumHiddenLayers, 
	                        dLayerWidth, relu );
	params = trueDregA.GetParamsVec();
	randomize_vector( params );
	trueDregA.SetParamsVec( params );

	ExpReg trueExpRegA( trueDregA );
	PSDReg truePsdRegA( trueLregA, trueExpRegA );
	PDReg truePdRegA( truePsdRegA, pdOffset );
	VectorType trueParamsA = truePdRegA.GetParamsVec();

	// B
	ConstantRegressor trueLregB = ConstantRegressor( MatrixType( lOutDim, 1 ) );
	params = trueLregB.GetParamsVec();
	randomize_vector( params );
	trueLregB.SetParamsVec( params );

	BaseRegressor trueDregB( dFeatDim, dOutDim, dNumHiddenLayers, 
	                         dLayerWidth, relu );
	params = trueDregB.GetParamsVec();
	randomize_vector( params );
	trueDregB.SetParamsVec( params );

	ExpReg trueExpRegB( trueDregB );
	PSDReg truePsdRegB( trueLregB, trueExpRegB );
	PDReg truePdRegB( truePsdRegB, pdOffset );
	VectorType trueParamsB = truePdRegB.GetParamsVec();

	// Initial model
	// A
	ConstantRegressor lregA( MatrixType::Zero( lOutDim, 1 ) );

	BaseRegressor dregA( dFeatDim, dOutDim, dNumHiddenLayers, 
	                    dLayerWidth, relu );
	params = dregA.GetParamsVec();
	randomize_vector( params );
	dregA.SetParamsVec( params );

	ExpReg expRegA( dregA );
	PSDReg psdRegA( lregA, dregA );
	PDReg pdRegA( psdRegA, pdOffset );

	// B
	ConstantRegressor lregB( MatrixType::Zero( lOutDim, 1 ) );

	BaseRegressor dregB( dFeatDim, dOutDim, dNumHiddenLayers, 
	                    dLayerWidth, relu );
	params = dregB.GetParamsVec();
	randomize_vector( params );
	dregB.SetParamsVec( params );

	ExpReg expRegB( dregB );
	PSDReg psdRegB( lregB, dregB );
	PDReg pdRegB( psdRegB, pdOffset );

	// Create test population
	unsigned int popSize = 1000;
	std::cout << "Sampling " << popSize << " datapoints..." << std::endl;
	
	std::vector<CovEst> estimatesA, estimatesB;
	std::vector<TransCovEst> transEstsA, transEstsB;
	std::vector<SumCovEst> sumEsts;
	std::vector<GLL> baseCosts;
	estimatesA.reserve( popSize );
	estimatesB.reserve( popSize );
	transEstsA.reserve( popSize );
	transEstsB.reserve( popSize );
	sumEsts.reserve( popSize );
	baseCosts.reserve( popSize );

	MultivariateGaussian<> mvg( MultivariateGaussian<>::VectorType::Zero( matDim ),
	                            MultivariateGaussian<>::MatrixType::Identity( matDim, matDim ) );

	for( unsigned int i = 0; i < popSize; i++ )
	{
		PDReg::InputType pdInputA;
		pdInputA.lInput = VectorType( lFeatDim );
		pdInputA.dInput = VectorType( dFeatDim );
		randomize_vector( pdInputA.lInput );
		randomize_vector( pdInputA.dInput );

		PDReg::InputType pdInputB;
		pdInputB.lInput = VectorType( lFeatDim );
		pdInputB.dInput = VectorType( dFeatDim );
		randomize_vector( pdInputB.lInput );
		randomize_vector( pdInputB.dInput );

		MatrixType transform = MatrixType::Random( matDim, matDim );
		MatrixType trueCovA = transform * truePdRegA.Evaluate( pdInputA ) * transform.transpose();
		MatrixType trueCovB = transform * truePdRegB.Evaluate( pdInputB ) * transform.transpose();

		mvg.SetCovariance( trueCovA + trueCovB );
		VectorType sample = mvg.Sample(); 

		estimatesA.emplace_back( pdRegA, pdInputA );
		estimatesB.emplace_back( pdRegB, pdInputB );
		transEstsA.emplace_back( estimatesA.back(), transform );
		transEstsB.emplace_back( estimatesB.back(), transform );
		sumEsts.emplace_back( estimatesA.back(), estimatesB.back() );
		baseCosts.emplace_back( sumEsts.back(), sample );
	}
	std::cout << "Sampling complete." << std::endl;

	MeanGLL meanCost( baseCosts );
	PenalizedMeanGLL penalizedMeanCosts( meanCost, 0 );

	unsigned int minibatchSize = 30;
	StochasticGLL stochasticCost( baseCosts, minibatchSize );
	PenalizedStochasticGLL penalizedStochasticCosts( stochasticCost, 1E-6 );

	VectorType initParams = sumEsts[0].GetParamsVec();
	VectorType trueParams( trueParamsA.size() + trueParamsB.size() );
	trueParams.head( trueParamsA.size() ) = trueParamsA;
	trueParams.tail( trueParamsB.size() ) = trueParamsB;

	// NLOptParameters optParams;
	// optParams.algorithm = nlopt::LD_LBFGS;
	// NLOptimizer nlOpt( optParams );
	// TestOptimization( nlOpt, penalizedMeanCosts, initParams, trueParams );

	AdamStepper stepper;
	SimpleConvergenceCriteria criteria;
	criteria.maxRuntime = 60;
	criteria.minElementGradient = 1E-3;
	criteria.minObjectiveDelta = 1E-3;
	SimpleConvergence convergence( criteria );

	AdamOptimizer modOpt( stepper, convergence );
	TestOptimization( modOpt, penalizedStochasticCosts, initParams, trueParams );
	double finalMeanObj = meanCost.Evaluate();
	meanCost.SetParamsVec( initParams );
	double initialMeanObj = meanCost.Evaluate();
	meanCost.SetParamsVec( trueParams );
	double trueMeanObj = meanCost.Evaluate();
	std::cout << "Initial mean objective: " << initialMeanObj << std::endl;
	std::cout << "Final mean objective: " << finalMeanObj << std::endl;
	std::cout << "True mean objective: " << trueMeanObj << std::endl;

}