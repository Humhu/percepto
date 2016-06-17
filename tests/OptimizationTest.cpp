#include "percepto/compo/ConstantRegressor.hpp"
#include "percepto/compo/ExponentialWrapper.hpp"
#include "percepto/compo/ModifiedCholeskyWrapper.hpp"
#include "percepto/compo/TransformWrapper.hpp"
#include "percepto/compo/AdditiveWrapper.hpp"
#include "percepto/compo/OffsetWrapper.hpp"

#include "percepto/utils/LowerTriangular.hpp"

#include "percepto/optim/ParameterL2Cost.hpp"
#include "percepto/optim/GaussianLogLikelihoodCost.hpp"
#include "percepto/optim/StochasticMeanCost.hpp"
#include "percepto/optim/OptimizerTypes.h"

#include "percepto/utils/MultivariateGaussian.hpp"
#include "percepto/utils/Randomization.hpp"

#include "percepto/neural/NetworkTypes.h"

#include <deque>
#include <ctime>
#include <iostream>

using namespace percepto;

typedef ReLUNet BaseRegressor;
typedef ExponentialWrapper<VectorType> ExpModule;
typedef ModifiedCholeskyWrapper PSDModule;
typedef OffsetWrapper<MatrixType> MatrixOffset;
typedef TransformWrapper TransCovEstimate;

unsigned int matDim = 6;
unsigned int dFeatDim = 5;
unsigned int lOutDim = matDim*(matDim-1)/2;
unsigned int dOutDim = matDim;

unsigned int numHiddenLayers = 1;
unsigned int layerWidth = 10;

double l2Weight = 0;
unsigned int minibatchSize = 30;

struct Regressor
{
	TerminalSource<VectorType> dInput;
	ConstantVectorRegressor lReg;
	ReLUNet dReg;
	ExpModule expReg;
	PSDModule psdReg;
	MatrixOffset pdReg;

	Regressor()
	: lReg( lOutDim ), 
	dReg( dFeatDim, dOutDim, numHiddenLayers, layerWidth,
	      HingeActivation( 1.0, 1E-3 ) )
	{
		dReg.SetSource( &dInput );
		expReg.SetSource( &dReg.GetOutputSource() );
		psdReg.SetLSource( &lReg );
		psdReg.SetDSource( &expReg );
		pdReg.SetSource( &psdReg );
		pdReg.SetOffset( 1E-9 * MatrixType::Identity( matDim, matDim ) );
	}

	Regressor( const Regressor& other )
	: lReg( other.lReg ), dReg( other.dReg )
	{
		dReg.SetSource( &dInput );
		expReg.SetSource( &dReg.GetOutputSource() );
		psdReg.SetLSource( &lReg );
		psdReg.SetDSource( &expReg );
		pdReg.SetSource( &psdReg );
		pdReg.SetOffset( 1E-9 * MatrixType::Identity( matDim, matDim ) );
	}

	void Invalidate()
	{
		lReg.Invalidate();
		dInput.Invalidate();
	}

	void Foreprop()
	{
		lReg.Foreprop();
		dInput.Foreprop();
	}

	void Backprop()
	{
		pdReg.Backprop( MatrixType() );
	}

};

struct Likelihood
{
	Regressor regA;
	Regressor regB;
	TransformWrapper transRegA;
	TransformWrapper transRegB;
	AdditiveWrapper<MatrixType> sumReg;
	GaussianLogLikelihoodCost gll;

	Likelihood()
	{
		transRegA.SetSource( &regA.pdReg );
		transRegB.SetSource( &regB.pdReg );
		sumReg.SetSourceA( &transRegA );
		sumReg.SetSourceB( &transRegB );
		gll.SetSource( &sumReg );
	}

	Likelihood( const Regressor& a, const Regressor& b )
	: regA( a ), regB( b )
	{
		transRegA.SetSource( &regA.pdReg );
		transRegB.SetSource( &regB.pdReg );
		sumReg.SetSourceA( &transRegA );
		sumReg.SetSourceB( &transRegB );
		gll.SetSource( &sumReg );
	}

	void Invalidate() 
	{
		regA.Invalidate(); 
		regB.Invalidate();
	}
	
	void Foreprop() 
	{
		regA.Foreprop();
		regB.Foreprop();
	}

	void Backprop() { gll.Backprop( MatrixType() ); }
};

struct OptimizationProblem
{
	std::deque<Likelihood> likelihoods;
	StochasticMeanCost<double> loss;
	ParameterL2Cost regularizer;
	AdditiveWrapper<double> objective;

	OptimizationProblem()
	{
		objective.SetSourceA( &regularizer );
		objective.SetSourceB( &loss );
	}

	double GetOutput() { return objective.GetOutput(); }

	void Invalidate()
	{
		for( unsigned int i = 0; i < likelihoods.size(); i++ )
		{
			likelihoods[i].Invalidate();
		}
		regularizer.Invalidate();
	}

	void Foreprop()
	{
		for( unsigned int i = 0; i < likelihoods.size(); i++ )
		{
			likelihoods[i].Foreprop();
		}
		regularizer.Foreprop();
	}

	void Backprop()
	{
		objective.Backprop( MatrixType() );
	}
};

template <class Optimizer, typename Problem>
void TestOptimization( Optimizer& opt, Problem& problem,
                       ParameterWrapper& para,
                       const VectorType& initParams,
                       const VectorType& trueParams )
{
	std::cout << "Beginning test with " << para.ParamDim() << " parameters." << std::endl;
	problem.Invalidate();
	problem.Foreprop();
	double initialObjective = problem.GetOutput();

	// opt.SetVerbosity( false ); // TODO
	OptimizationResults results;
	try
	{
		para.SetParamsVec( initParams );
		results = opt.Optimize( problem );
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
	problem.Invalidate();
	problem.Foreprop();
	double minCost = problem.GetOutput();

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

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "D feature dim: " << dFeatDim << std::endl;
	std::cout << "L output dim: " << lOutDim << std::endl;
	std::cout << "D output dim: " << dOutDim << std::endl;

	MatrixType pdOffset = 1E-2 * MatrixType::Identity( matDim, matDim );
	HingeActivation relu( 1.0, 1E-3 );

	Regressor trueRegA, trueRegB, regA, regB;
	VectorType p;
	Parameters::Ptr temp;

	// True A
	Parameters::Ptr trueLParamsA = trueRegA.lReg.CreateParameters();
	Parameters::Ptr trueDParamsA = trueRegA.dReg.CreateParameters();
	// True B
	Parameters::Ptr trueLParamsB = trueRegB.lReg.CreateParameters();
	Parameters::Ptr trueDParamsB = trueRegB.dReg.CreateParameters();

	ParameterWrapper trueParams;
	trueParams.AddParameters( trueLParamsA );
	trueParams.AddParameters( trueDParamsA );
	trueParams.AddParameters( trueLParamsB );
	trueParams.AddParameters( trueDParamsB );
	
	p = VectorType( trueParams.ParamDim() );
	randomize_vector( p );
	trueParams.SetParamsVec( p );

	// Init
	Parameters::Ptr lParamsA = regA.lReg.CreateParameters();
	Parameters::Ptr dParamsA = regA.dReg.CreateParameters();
	
	Parameters::Ptr lParamsB = regB.lReg.CreateParameters();
	Parameters::Ptr dParamsB = regB.dReg.CreateParameters();

	ParameterWrapper params;
	params.AddParameters( lParamsA );
	params.AddParameters( dParamsA );
	params.AddParameters( lParamsB );
	params.AddParameters( dParamsB );

	p = VectorType( params.ParamDim() );
	randomize_vector( p );
	params.SetParamsVec( p );

	ParameterL2Cost l2Cost;
	l2Cost.SetParameters( &params );
	l2Cost.SetWeight( l2Weight );

	// Create test population
	unsigned int popSize = 1000;
	std::cout << "Sampling " << popSize << " datapoints..." << std::endl;
	
	OptimizationProblem problem;

	MultivariateGaussian<> mvg( MultivariateGaussian<>::VectorType::Zero( matDim ),
	                            MultivariateGaussian<>::MatrixType::Identity( matDim, matDim ) );

	problem.regularizer.SetParameters( &params );
	problem.loss.SetBatchSize( minibatchSize );
	for( unsigned int i = 0; i < popSize; i++ )
	{
		VectorType dInputA( dFeatDim );
		randomize_vector( dInputA );
		VectorType dInputB( dFeatDim );
		randomize_vector( dInputB );

		trueRegA.dInput.SetOutput( dInputA );
		trueRegB.dInput.SetOutput( dInputB );
		trueRegA.Invalidate();
		trueRegA.Foreprop();
		trueRegB.Invalidate();
		trueRegB.Foreprop();
		MatrixType outputA = trueRegA.pdReg.GetOutput();
		MatrixType outputB = trueRegB.pdReg.GetOutput();

		MatrixType transformA = MatrixType::Random( matDim, matDim );
		MatrixType transformB = MatrixType::Random( matDim, matDim );
		MatrixType trueCovA = transformA * outputA * transformA.transpose();
		MatrixType trueCovB = transformB * outputB * transformB.transpose();

		mvg.SetCovariance( trueCovA + trueCovB );
		VectorType sample = mvg.Sample(); 

		problem.likelihoods.emplace_back( regA, regB );
		problem.likelihoods[i].regA.dInput.SetOutput( dInputA );
		problem.likelihoods[i].regB.dInput.SetOutput( dInputB );
		problem.likelihoods[i].transRegA.SetTransform( transformA );
		problem.likelihoods[i].transRegB.SetTransform( transformB );
		problem.likelihoods[i].gll.SetSample( sample );
		problem.loss.AddSource( &problem.likelihoods[i].gll );
	}
	std::cout << "Sampling complete." << std::endl;

	VectorType initParamsVec = params.GetParamsVec();
	VectorType trueParamsVec = trueParams.GetParamsVec();

	// NLOptParameters optParams;
	// optParams.algorithm = nlopt::LD_LBFGS;
	// NLOptimizer nlOpt( optParams );
	// TestOptimization( nlOpt, penalizedMeanCosts, initParams, trueParams );

	AdamStepper stepper;
	SimpleConvergenceCriteria criteria;
	criteria.maxRuntime = 120;
	criteria.minElementGradient = 1E-3;
	SimpleConvergence convergence( criteria );

	AdamOptimizer modOpt( stepper, convergence, params );
	TestOptimization( modOpt, problem, params, 
	                  initParamsVec, trueParamsVec );
	
	problem.Invalidate();
	problem.Foreprop();
	problem.loss.ParentCost::Foreprop();
	double finalMeanObj = problem.loss.ParentCost::GetOutput();
	
	params.SetParamsVec( initParamsVec );
	problem.Invalidate();
	problem.Foreprop();
	problem.loss.ParentCost::Foreprop();
	double initialMeanObj = problem.loss.ParentCost::GetOutput();

	params.SetParamsVec( trueParamsVec );
	problem.Invalidate();
	problem.Foreprop();
	problem.loss.ParentCost::Foreprop();
	double trueMeanObj = problem.loss.ParentCost::GetOutput();

	std::cout << "Initial mean objective: " << initialMeanObj << std::endl;
	std::cout << "Final mean objective: " << finalMeanObj << std::endl;
	std::cout << "True mean objective: " << trueMeanObj << std::endl;

}
