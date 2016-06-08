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
typedef InputWrapper<BaseRegressor> BaseModule;
typedef ExponentialWrapper<BaseModule> ExpModule;
typedef ModifiedCholeskyWrapper<ConstantRegressor, ExpModule> PSDModule;
typedef OffsetWrapper<PSDModule> PDModule;
typedef InputChainWrapper<BaseModule,PDModule> CovEstimator;

typedef InputWrapper<CovEstimator> CovEstimate;
typedef TransformWrapper<CovEstimate> TransCovEstimate;
typedef AdditiveWrapper<CovEstimate,CovEstimate> SumCovEstimate;
typedef GaussianLogLikelihoodCost<SumCovEstimate> GLL;

typedef MeanPopulationCost<GLL> MeanGLL;
typedef StochasticPopulationCost<GLL> StochasticGLL;
typedef AdditiveWrapper<MeanGLL,ParameterL2Cost> PenalizedMeanGLL;
typedef AdditiveWrapper<StochasticGLL,ParameterL2Cost> PenalizedStochasticGLL;

typedef NLOptInterface NLOptimizer;

template <class Optimizer, typename Cost>
void TestOptimization( Optimizer& opt, Cost& cost,
                       ParametricWrapper& para,
                       const VectorType& initParams,
                       const VectorType& trueParams )
{
	std::cout << "Beginning test with " << para.ParamDim() << " parameters." << std::endl;
	double initialObjective = cost.Evaluate();

	// opt.SetVerbosity( false );
	OptimizationResults results;
	try
	{
		para.SetParamsVec( initParams );
		results = opt.Optimize( cost );
	}
	catch( std::runtime_error e )
	{
		std::cout << "Received exception: " << e.what() << std::endl;
		return;
	}

	VectorType finalParams = para.GetParamsVec();
	VectorType delta = trueParams - finalParams;
	double errorNorm = delta.norm() / delta.size();
	double errorMax = std::max( -delta.minCoeff(), delta.maxCoeff() );

	para.SetParamsVec( trueParams );
	double minCost = cost.Evaluate();

	para.SetParamsVec( finalParams );

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
	unsigned int dFeatDim = 5;
	unsigned int lOutDim = TriangularMapping::num_positions( matDim - 1 );
	unsigned int dOutDim = matDim;

	unsigned int dNumHiddenLayers = 1;
	unsigned int dLayerWidth = 10;

	double l2Weight = 0;

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "D feature dim: " << dFeatDim << std::endl;
	std::cout << "L output dim: " << lOutDim << std::endl;
	std::cout << "D output dim: " << dOutDim << std::endl;

	MatrixType pdOffset = 1E-2 * MatrixType::Identity( matDim, matDim );
	HingeActivation relu( 1.0, 1E-3 );

	// True model
	// A
	ConstantRegressor trueLRegA = ConstantRegressor( MatrixType( lOutDim, 1 ) );
	VectorType params = trueLRegA.GetParamsVec();
	randomize_vector( params );
	trueLRegA.SetParamsVec( params );

	BaseRegressor trueDregA( dFeatDim, dOutDim, dNumHiddenLayers, 
	                        dLayerWidth, relu );
	params = trueDregA.GetParamsVec();
	randomize_vector( params );
	trueDregA.SetParamsVec( params );

	BaseModule trueDregWrapperA( trueDregA );
	ExpModule trueExpModuleA( trueDregWrapperA );
	PSDModule truePsdRegA( trueLRegA, trueExpModuleA );
	PDModule truePdRegA( truePsdRegA, pdOffset );
	CovEstimator trueEstimatorA( trueDregWrapperA, truePdRegA );

	// B
	ConstantRegressor trueLRegB = ConstantRegressor( MatrixType( lOutDim, 1 ) );
	params = trueLRegB.GetParamsVec();
	randomize_vector( params );
	trueLRegB.SetParamsVec( params );

	BaseRegressor trueDregB( dFeatDim, dOutDim, dNumHiddenLayers, 
	                         dLayerWidth, relu );
	params = trueDregB.GetParamsVec();
	randomize_vector( params );
	trueDregB.SetParamsVec( params );

	BaseModule trueDregWrapperB( trueDregB );
	ExpModule trueExpModuleB( trueDregWrapperB );
	PSDModule truePsdRegB( trueLRegB, trueExpModuleB );
	PDModule truePdRegB( truePsdRegB, pdOffset );
	CovEstimator trueEstimatorB( trueDregWrapperB, truePdRegB );

	ParametricWrapper trueParametric;
	trueParametric.AddParametric( &trueLRegA );
	trueParametric.AddParametric( &trueDregA );
	trueParametric.AddParametric( &trueLRegB );
	trueParametric.AddParametric( &trueDregB );

	// Initial model
	// A
	ConstantRegressor lRegA( MatrixType::Zero( lOutDim, 1 ) );

	BaseRegressor dRegA( dFeatDim, dOutDim, dNumHiddenLayers, 
	                    dLayerWidth, relu );
	params = dRegA.GetParamsVec();
	randomize_vector( params );
	dRegA.SetParamsVec( params );
	BaseModule dRegWrapperA( dRegA );
	ExpModule expRegA( dRegWrapperA );
	PSDModule psdRegA( lRegA, expRegA );
	PDModule pdRegA( psdRegA, pdOffset );
	CovEstimator estimatorA( dRegWrapperA, pdRegA );

	ParametricWrapper parametricA;
	parametricA.AddParametric( &lRegA );
	parametricA.AddParametric( &dRegA );

	// B
	ConstantRegressor lRegB( MatrixType::Zero( lOutDim, 1 ) );

	BaseRegressor dRegB( dFeatDim, dOutDim, dNumHiddenLayers, 
	                    dLayerWidth, relu );
	params = dRegB.GetParamsVec();
	randomize_vector( params );
	dRegB.SetParamsVec( params );
	BaseModule dRegWrapperB( dRegB );
	ExpModule expRegB( dRegWrapperB );
	PSDModule psdRegB( lRegB, expRegB );
	PDModule pdRegB( psdRegB, pdOffset );
	CovEstimator estimatorB( dRegWrapperB, pdRegB );

	ParametricWrapper parametricB;
	parametricB.AddParametric( &lRegB );
	parametricB.AddParametric( &dRegB );

	ParametricWrapper jointParametric;
	jointParametric.AddParametric( &parametricA );
	jointParametric.AddParametric( &parametricB );
	ParameterL2Cost l2Cost( jointParametric, l2Weight );

	// Create test population
	unsigned int popSize = 1000;
	std::cout << "Sampling " << popSize << " datapoints..." << std::endl;
	
	std::vector<CovEstimate> estimatesA, estimatesB;
	std::vector<TransCovEstimate> transEstsA, transEstsB;
	std::vector<SumCovEstimate> sumEsts;
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
		VectorType dInputA( dFeatDim );
		randomize_vector( dInputA );
		VectorType dInputB( dFeatDim );
		randomize_vector( dInputB );

		MatrixType transform = MatrixType::Random( matDim, matDim );
		MatrixType trueCovA = transform * trueEstimatorA.Evaluate( dInputA ) * transform.transpose();
		MatrixType trueCovB = transform * trueEstimatorB.Evaluate( dInputB ) * transform.transpose();

		mvg.SetCovariance( trueCovA + trueCovB );
		VectorType sample = mvg.Sample(); 

		estimatesA.emplace_back( estimatorA, dInputA );
		estimatesB.emplace_back( estimatorB, dInputB );
		transEstsA.emplace_back( estimatesA.back(), transform );
		transEstsB.emplace_back( estimatesB.back(), transform );
		sumEsts.emplace_back( estimatesA.back(), estimatesB.back() );
		baseCosts.emplace_back( sumEsts.back(), sample );
	}
	std::cout << "Sampling complete." << std::endl;

	MeanGLL meanCost( baseCosts );
	PenalizedMeanGLL penalizedMeanCosts( meanCost, l2Cost );

	unsigned int minibatchSize = 30;
	StochasticGLL stochasticCost( baseCosts, minibatchSize );
	PenalizedStochasticGLL penalizedStochasticCosts( stochasticCost, l2Cost );

	VectorType initParams = jointParametric.GetParamsVec();
	VectorType trueParams = trueParametric.GetParamsVec();

	// NLOptParameters optParams;
	// optParams.algorithm = nlopt::LD_LBFGS;
	// NLOptimizer nlOpt( optParams );
	// TestOptimization( nlOpt, penalizedMeanCosts, initParams, trueParams );

	AdamStepper stepper;
	SimpleConvergenceCriteria criteria;
	criteria.maxRuntime = 120;
	criteria.minElementGradient = 1E-3;
	criteria.minObjectiveDelta = 1E-3;
	SimpleConvergence convergence( criteria );

	AdamOptimizer modOpt( stepper, convergence, jointParametric );
	TestOptimization( modOpt, penalizedStochasticCosts, jointParametric, 
	                  initParams, trueParams );
	double finalMeanObj = meanCost.Evaluate();
	jointParametric.SetParamsVec( initParams );
	double initialMeanObj = meanCost.Evaluate();
	jointParametric.SetParamsVec( trueParams );
	double trueMeanObj = meanCost.Evaluate();
	std::cout << "Initial mean objective: " << initialMeanObj << std::endl;
	std::cout << "Final mean objective: " << finalMeanObj << std::endl;
	std::cout << "True mean objective: " << trueMeanObj << std::endl;

}