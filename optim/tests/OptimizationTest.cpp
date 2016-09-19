#include "modprop/compo/ConstantRegressor.hpp"
#include "modprop/compo/ExponentialWrapper.hpp"
#include "modprop/compo/ModifiedCholeskyWrapper.hpp"
#include "modprop/compo/TransformWrapper.hpp"
#include "modprop/compo/AdditiveWrapper.hpp"
#include "modprop/compo/OffsetWrapper.hpp"

#include "modprop/utils/LowerTriangular.hpp"

#include "modprop/optim/ParameterL2Cost.hpp"
#include "modprop/optim/GaussianLogLikelihoodCost.hpp"
#include "modprop/optim/StochasticMeanCost.hpp"

#include "optim/Optimizers.h"

#include "modprop/utils/MultivariateGaussian.hpp"
#include "modprop/utils/Randomization.hpp"

#include "modprop/neural/NetworkTypes.h"

#include <deque>
#include <ctime>
#include <iostream>

using namespace percepto;

typedef ReLUNet BaseRegressor;
typedef ExponentialWrapper ExpModule;
typedef ModifiedCholeskyWrapper PSDModule;
typedef OffsetWrapper<MatrixType> MatrixOffset;
typedef TransformWrapper TransCovEstimate;

unsigned int matDim = 6;
unsigned int dFeatDim = 5;
unsigned int lOutDim = matDim*(matDim-1)/2;
unsigned int dOutDim = matDim;

unsigned int numHiddenLayers = 1;
unsigned int layerWidth = 10;

double l2Weight = 1E-3;
unsigned int batchSize = 30;

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
	Regressor reg;
	TransformWrapper transReg;
	GaussianLogLikelihoodCost gll;

	Likelihood( const Regressor& a )
	: reg( a )
	{
		transReg.SetSource( &reg.pdReg );
		gll.SetSource( &transReg );
	}

	void Invalidate() 
	{
		reg.Invalidate(); 
	}
	
	void Foreprop() 
	{
		reg.Foreprop();
	}

	void Backprop() { gll.Backprop( MatrixType() ); }
};

struct LikelihoodProblem
: public NaturalOptimizationProblem
{
	std::deque<Likelihood> likelihoods;
	StochasticMeanCost<double> loss;
	ParameterL2Cost regularizer;
	AdditiveWrapper<double> objective;

	Parameters::Ptr params;

	LikelihoodProblem( Parameters::Ptr p,
	                   unsigned int batchSize,
	                   double l2Weight )
	{
		params = p;
		loss.SetBatchSize( batchSize );
		regularizer.SetWeight( l2Weight );
		regularizer.SetParameters( params );
		objective.SetSourceA( &regularizer );
		objective.SetSourceB( &loss );
	}

	virtual bool IsMinimization() const { return false; }

	virtual void Resample()
	{
		loss.Resample();
	}

	virtual double ComputeObjective()
	{
		Invalidate();
		Foreprop();
		return objective.GetOutput();
	}

	virtual VectorType ComputeGradient()
	{
		Invalidate();
		Foreprop();
		Backprop();
		return params->GetDerivs();
	}

	virtual VectorType ComputeNaturalGradient()
	{
		Invalidate();
		Foreprop();
		BackpropNatural();
		return params->GetDerivs();
	}

	virtual VectorType GetParameters() const
	{
		return params->GetParamsVec();
	}

	virtual void SetParameters( const VectorType& p )
	{
		params->SetParamsVec( p );
	}

	void Invalidate()
	{
		for( unsigned int i = 0; i < likelihoods.size(); i++ )
		{
			likelihoods[i].Invalidate();
		}
		regularizer.Invalidate();
		params->ResetAccumulators();
	}

	void Foreprop()
	{
		const std::vector<unsigned int>& inds = loss.GetActiveInds();
		BOOST_FOREACH( unsigned int i, inds )
		{
			likelihoods[i].Foreprop();
		}
		regularizer.Foreprop();
	}

	void ForepropAll()
	{
		for( unsigned int i = 0; i < likelihoods.size(); i++ )
		{
			likelihoods[i].Foreprop();
		}
		regularizer.Foreprop();
	}

	void Backprop()
	{
		objective.Backprop( MatrixType::Identity(1,1) );
	}

	void BackpropNatural()
	{
		const std::vector<unsigned int>& inds = loss.GetActiveInds();
		MatrixType dodw = MatrixType::Identity( 1, 1 ) / inds.size();
		BOOST_FOREACH( unsigned int i, inds )
		{
			likelihoods[i].gll.Backprop( dodw );
		}
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
	problem.Resample();

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
	double minCost = problem.ComputeObjective();

	para.SetParamsVec( finalParams );

	std::cout << "True params: " << std::endl << trueParams.transpose() << std::endl;
	std::cout << "Final params: " << std::endl << finalParams.transpose() << std::endl;
	std::cout << "\tInitial objective: " << results.initialObjective << std::endl;
	std::cout << "\tFinal objective: " << results.finalObjective << std::endl;
	std::cout << "\tTrue objective: " << minCost << std::endl;
	// std::cout << "\tOverall time: " << results.totalElapsedSecs << std::endl;
	// std::cout << "\tObjective time: " << results.totalObjectiveSecs << std::endl;
	// std::cout << "\tGradient time: " << results.totalGradientSecs << std::endl;
	// std::cout << "\tObjective evaluations: " << results.numObjectiveEvaluations << std::endl;
	// std::cout << "\tGradient evaluations: " << results.numGradientEvaluations << std::endl;
	// std::cout << "\tAvg time/objective: " << results.totalObjectiveSecs/results.numObjectiveEvaluations << std::endl;
	// std::cout << "\tAvg time/gradient: " << results.totalGradientSecs/results.numGradientEvaluations << std::endl;
	std::cout << "\tAverage error norm: " << errorNorm << std::endl;
	std::cout << "\tMax error: " << errorMax << std::endl;
}

int main( void )
{
	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "D feature dim: " << dFeatDim << std::endl;
	std::cout << "L output dim: " << lOutDim << std::endl;
	std::cout << "D output dim: " << dOutDim << std::endl;

	Regressor trueRegA, reg;
	VectorType p;
	Parameters::Ptr temp;

	// True A
	Parameters::Ptr trueLParamsA = trueRegA.lReg.CreateParameters();
	Parameters::Ptr trueDParamsA = trueRegA.dReg.CreateParameters();

	ParameterWrapper trueParams;
	trueParams.AddParameters( trueLParamsA );
	trueParams.AddParameters( trueDParamsA );
	
	p = VectorType( trueParams.ParamDim() );
	randomize_vector( p, -0.5, 0.5 );
	trueParams.SetParamsVec( p );

	// Init
	Parameters::Ptr lParamsA = reg.lReg.CreateParameters();
	Parameters::Ptr dParamsA = reg.dReg.CreateParameters();
	
	ParameterWrapper::Ptr params = std::make_shared<ParameterWrapper>();
	params->AddParameters( lParamsA );
	params->AddParameters( dParamsA );

	p = VectorType( params->ParamDim() );
	randomize_vector( p, -0.2, 0.2 );
	params->SetParamsVec( p );

	// Create test population
	unsigned int popSize = 1000;
	std::cout << "Sampling " << popSize << " datapoints..." << std::endl;
	

	LikelihoodProblem problem( params, batchSize, l2Weight );

	MultivariateGaussian<> mvg( MultivariateGaussian<>::VectorType::Zero( matDim ),
	                            MultivariateGaussian<>::MatrixType::Identity( matDim, matDim ) );

	for( unsigned int i = 0; i < popSize; i++ )
	{
		VectorType dInputA( dFeatDim );
		randomize_vector( dInputA );

		trueRegA.dInput.SetOutput( dInputA );
		trueRegA.Invalidate();
		trueRegA.Foreprop();
		MatrixType outputA = trueRegA.pdReg.GetOutput();

		MatrixType transformA = MatrixType::Random( matDim, matDim );
		MatrixType trueCovA = transformA * outputA * transformA.transpose();

		mvg.SetCovariance( trueCovA );
		VectorType sample = mvg.Sample(); 

		problem.likelihoods.emplace_back( reg );
		problem.likelihoods[i].reg.dInput.SetOutput( dInputA );
		problem.likelihoods[i].transReg.SetTransform( transformA );
		problem.likelihoods[i].gll.SetSample( sample );
		problem.loss.AddSource( &problem.likelihoods[i].gll );
	}
	std::cout << "Sampling complete." << std::endl;

	VectorType initParamsVec = params->GetParamsVec();
	VectorType trueParamsVec = trueParams.GetParamsVec();

	ModularOptimizer optimizer;
	
	// AdamSearchDirector::Ptr director = std::make_shared<AdamSearchDirector>();
	// GradientSearchDirector::Ptr director = std::make_shared<GradientSearchDirector>();
	NaturalSearchDirector::Ptr director = std::make_shared<NaturalSearchDirector>();
	optimizer.SetSearchDirector( director );

	BacktrackingSearchStepper::Ptr stepper = std::make_shared<BacktrackingSearchStepper>();
	stepper->SetInitialStep( 1E-3 );
	stepper->SetBacktrackingRatio( 0.5 );
	stepper->SetMaxBacktracks( 20 );
	stepper->SetImprovementRatio( 0.75 );

	optimizer.SetSearchStepper( stepper );

	RuntimeTerminationChecker::Ptr runtimeChecker = std::make_shared<RuntimeTerminationChecker>();
	runtimeChecker->SetMaxRuntime( 20 );
	optimizer.AddTerminationChecker( runtimeChecker );

	TestOptimization( optimizer, problem, *params, 
	                  initParamsVec, trueParamsVec );
	
	problem.Invalidate();
	problem.Foreprop();
	problem.loss.ParentCost::Foreprop();
	double finalMeanObj = problem.loss.ParentCost::GetOutput();
	
	params->SetParamsVec( initParamsVec );
	problem.Invalidate();
	problem.Foreprop();
	problem.loss.ParentCost::Foreprop();
	double initialMeanObj = problem.loss.ParentCost::GetOutput();

	params->SetParamsVec( trueParamsVec );
	problem.Invalidate();
	problem.Foreprop();
	problem.loss.ParentCost::Foreprop();
	double trueMeanObj = problem.loss.ParentCost::GetOutput();

	std::cout << "Initial mean objective: " << initialMeanObj << std::endl;
	std::cout << "Final mean objective: " << finalMeanObj << std::endl;
	std::cout << "True mean objective: " << trueMeanObj << std::endl;

}
