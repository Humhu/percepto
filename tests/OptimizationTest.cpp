#include "percepto/LinearRegressor.hpp"
#include "percepto/ExponentialWrapper.hpp"
#include "percepto/AffineWrapper.hpp"

#include "percepto/pdreg/LowerTriangular.hpp"
#include "percepto/pdreg/ModifiedCholeskyWrapper.hpp"

#include "percepto/GaussianLogLikelihoodCost.hpp"
#include "percepto/MeanPopulationCost.hpp"
#include "percepto/ParameterL2Cost.hpp"

#include "percepto/optim/NlOptInterface.hpp"

#include "percepto/utils/MultivariateGaussian.hpp"

#include <ctime>
#include <iostream>

using namespace percepto;

typedef LinearRegressor LinReg;
typedef ExponentialWrapper<LinearRegressor> ExpReg;
typedef ModifiedCholeskyWrapper<LinearRegressor,ExpReg> ModCholReg;

typedef AffineWrapper<ModCholReg> AffineModCholReg;

typedef GaussianLogLikelihoodCost<AffineModCholReg> GLL;
typedef MeanPopulationCost<GLL> MeanGLL;
typedef ParameterL2Cost<MeanGLL> MeanGLL_L2;

typedef NLOptInterface<MeanGLL_L2> NLOptimizer;

template <class Optimizer>
void TestOptimization( Optimizer& opt,
                       const VectorType& initParams,
                       const VectorType& trueParams )
{
	std::cout << "Beginning test of " << opt.GetAlgorithmName()
	          << " with " << opt.GetCost().ParamDim() << " parameters... " << std::endl;

	opt.SetVerbosity( false );
	typename Optimizer::ResultsType results;
	try
	{
		opt.GetCost().SetParamsVec( initParams );
		results = opt.Optimize();
	}
	catch( std::runtime_error e )
	{
		std::cout << "Received exception: " << e.what() << std::endl;
		return;
	}

	VectorType finalParams = opt.GetCost().GetParamsVec();
	VectorType delta = trueParams - finalParams;
	double errorNorm = delta.norm() / delta.size();
	double errorMax = std::max( -delta.minCoeff(), delta.maxCoeff() );

	opt.GetCost().SetParamsVec( trueParams );
	double minCost = opt.GetCost().Evaluate();

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

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "L feature dim: " << lFeatDim << std::endl;
	std::cout << "D feature dim: " << dFeatDim << std::endl;
	std::cout << "L output dim: " << lOutDim << std::endl;
	std::cout << "D output dim: " << dOutDim << std::endl;

	MatrixType mcOffset = 1E-2 * MatrixType::Identity( matDim, matDim );

	// True model
	LinReg trueLinreg( LinReg::ParamType::Random( lOutDim, lFeatDim ) );
	ExpReg trueExpreg( ExpReg::ParamType::Random( dOutDim, dFeatDim ) );
	ModCholReg trueMc( trueLinreg, trueExpreg, mcOffset );
	AffineModCholReg trueModel( trueMc );
	VectorType trueParams = trueMc.GetParamsVec();

	// Test weight creation
	VectorType weights = trueMc.CreateWeightVector( 0.1, 0.2 );
	std::cout << "weights: " << weights.transpose() << std::endl;

	// Initial model
	ModCholReg mcReg( ModCholReg::create_zeros( lFeatDim, dFeatDim, matDim ),
	                   mcOffset );
	AffineModCholReg amcReg( mcReg );

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
		baseCosts.emplace_back( amcInput, sample, amcReg );
	}
	std::cout << "Sampling complete." << std::endl;

	MeanGLL meanCost( baseCosts );
	MeanGLL_L2 finalCost( meanCost, 1E-3 );

	VectorType initParams = amcReg.GetParamsVec();

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
		NLOptimizer opt( finalCost, optParams );
		TestOptimization( opt, initParams, trueParams );
	}

}