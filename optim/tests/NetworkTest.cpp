#include "modprop/compo/AdditiveWrapper.hpp"
#include "modprop/neural/LinearLayer.hpp"
#include "modprop/neural/FullyConnectedNet.hpp"

#include "modprop/neural/HingeActivation.hpp"
#include "modprop/neural/SigmoidActivation.hpp"
#include "modprop/neural/NullActivation.hpp"
#include "modprop/neural/NetworkTypes.h"

#include "modprop/optim/SquaredLoss.hpp"
#include "modprop/optim/StochasticMeanCost.hpp"
#include "modprop/optim/ParameterL2Cost.hpp"

#include "optim/Optimizers.h"

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "modprop/utils/Derivatives.hpp"
#include "modprop/utils/Randomization.hpp"

#include <deque>
#include <cstdlib>
#include <iostream>

using namespace percepto;

// Comment the above and uncomment below to use Rectified Linear Units instead
typedef PerceptronNet TestNet;
// typedef ReLUNet TestNet;

unsigned int inputDim = 1;
unsigned int outputDim = 1;
unsigned int numHiddenLayers = 3;
unsigned int layerWidth = 20;
unsigned int batchSize = 20;

struct Regressor
{
	TerminalSource<VectorType> netInput;
	TestNet net;
	SquaredLoss<VectorType> loss;

	Regressor()
	: net( inputDim, outputDim, numHiddenLayers, layerWidth,
	       SigmoidActivation(), TestNet::OUTPUT_UNRECTIFIED )
	{
		net.SetSource( &netInput );
		loss.SetSource( &net.GetOutputSource() );
	}

	Regressor( const Regressor& other )
	: net( other.net )
	{
		net.SetSource( &netInput );
		loss.SetSource( &net.GetOutputSource() );
	}

	void Foreprop()
	{
		netInput.Foreprop();
	}

	double GetOutput()
	{
		return net.GetOutput()(0);
	}

	void Invalidate()
	{
		netInput.Invalidate();
	}
};

struct RegressionProblem
: public NaturalOptimizationProblem
{
	std::deque<Regressor> regressors;
	StochasticMeanCost<double> losses;
	ParameterL2Cost regularizer;
	AdditiveWrapper<double> objective;
	
	Parameters::Ptr params;

	RegressionProblem( Parameters::Ptr p,
	                   unsigned int batchSize,
	                   double l2Weight ) 
	{
		params = p;
		losses.SetBatchSize( batchSize );
		regularizer.SetWeight( l2Weight );
		regularizer.SetParameters( params );
		objective.SetSourceA( &losses );
		objective.SetSourceB( &regularizer );
	}

	virtual bool IsMinimization() const { return true; }

	virtual void Resample()
	{
		losses.Resample();
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
		for( unsigned int i = 0; i < regressors.size(); i++ )
		{
			regressors[i].Invalidate();
		}
		regularizer.Invalidate();
		params->ResetAccumulators();
	}

	void Foreprop()
	{
		const std::vector<unsigned int>& inds = losses.GetActiveInds();
		BOOST_FOREACH( unsigned int i, inds )
		{
			regressors[i].Foreprop();
		}
		regularizer.Foreprop();
	}

	void ForepropAll()
	{
		for( unsigned int i = 0; i < regressors.size(); i++ )
		{
			regressors[i].Foreprop();
		}
		regularizer.Foreprop();
	}

	void Backprop()
	{
		objective.Backprop( MatrixType::Identity(1,1) );
	}

	void BackpropNatural()
	{
		const std::vector<unsigned int>& inds = losses.GetActiveInds();
		MatrixType dodw = MatrixType::Identity( 1, 1 ) / inds.size();
		BOOST_FOREACH( unsigned int i, inds )
		{
			regressors[i].loss.Backprop( dodw );
		}
	}
};

// The highly nonlinear function we will try to fit
double f( double x )
{
	return 8 * std::cos( x ) + 2.5 * x * sin( x ) + 2.8 * x;
}

// Generate a specified number of random xs and corresponding ys
void generate_data( std::vector<VectorType>& xs,
                    std::vector<VectorType>& ys,
                    unsigned int num_data )
{
	boost::random::mt19937 generator;
	boost::random::random_device rng;
	generator.seed( rng );
	boost::random::uniform_real_distribution<> xDist( -5.0, 5.0 );

	xs.clear(); 
	ys.clear(); 
	xs.reserve( num_data );
	ys.reserve( num_data );

	VectorType x(1,1);
	VectorType y(1,1);
	for( unsigned int i = 0; i < num_data; i++ )
	{
		x(0) = xDist( generator );
		y(0) = f( x(0) );
		xs.push_back( x );
		ys.push_back( y );
	}
}

int main( int argc, char** argv )
{
	unsigned int numTrain = 150;
	unsigned int numTest = 200;
	double l2Weight = 1E-3;

	std::vector<VectorType> xTest, yTest, xTrain, yTrain;
	generate_data( xTest, yTest, numTest );
	generate_data( xTrain, yTrain, numTrain );

	std::cout << "Initializing net..." << std::endl;
	std::cout << "Creating linear layers..." << std::endl;

	// // ReLU initialization
	Regressor reg;
	Parameters::Ptr params = reg.net.CreateParameters();

	// Randomize parameters
	VectorType p( params->ParamDim() );
	randomize_vector( p, -0.2, 0.2 );
	params->SetParamsVec( p );
	std::cout << "Initial net: " << std::endl << reg.net << std::endl;

	// Create the loss functions
	std::cout << "Generating losses..." << std::endl;
	RegressionProblem trainProblem( params, batchSize, l2Weight );
	RegressionProblem testProblem( params, batchSize, l2Weight );
	
	// NOTE If we don't reserve, the vector resizing and moving may
	// invalidate the references. Alternatively we can use a deque
	for( unsigned int i = 0; i < numTrain; i++ )
	{
		trainProblem.regressors.emplace_back( reg );
		trainProblem.regressors[i].netInput.SetOutput( xTrain[i] );
		trainProblem.regressors[i].loss.SetTarget( yTrain[i] );
		trainProblem.losses.AddSource( &trainProblem.regressors[i].loss );
	}

	for( unsigned int i = 0; i < numTest; i++ )
	{
		testProblem.regressors.emplace_back( reg );
		testProblem.regressors[i].netInput.SetOutput( xTest[i] );
		testProblem.regressors[i].loss.SetTarget( yTest[i] );
		testProblem.losses.AddSource( &testProblem.regressors[i].loss );
	}

	ModularOptimizer optimizer;
	
	// AdamSearchDirector::Ptr director = std::make_shared<AdamSearchDirector>();
	// GradientSearchDirector::Ptr director = std::make_shared<GradientSearchDirector>();
	NaturalSearchDirector::Ptr director = std::make_shared<NaturalSearchDirector>();
	optimizer.SetSearchDirector( director );

	BacktrackingSearchStepper::Ptr stepper = std::make_shared<BacktrackingSearchStepper>();
	stepper->SetInitialStep( 1E-1 );
	stepper->SetBacktrackingRatio( 0.5 );
	stepper->SetMaxBacktracks( 20 );
	stepper->SetImprovementRatio( 0.75 );

	// L1ConstrainedSearchStepper::Ptr stepper = std::make_shared<L1ConstrainedSearchStepper>();
	// stepper->SetMaxL1Norm( 1E-1 );
	// stepper->SetStepSize( 1E-2 );
	optimizer.SetSearchStepper( stepper );

	RuntimeTerminationChecker::Ptr runtimeChecker = std::make_shared<RuntimeTerminationChecker>();
	runtimeChecker->SetMaxRuntime( 20 );
	optimizer.AddTerminationChecker( runtimeChecker );

	// GradientTerminationChecker::Ptr gradientChecker = std::make_shared<GradientTerminationChecker>();
	// gradientChecker->SetMinGradientNorm( 1E-9 );
	// optimizer.AddTerminationChecker( gradientChecker );

	// IterationTerminationChecker::Ptr iterationChecker = std::make_shared<IterationTerminationChecker>();
	// iterationChecker->SetMaxIterations( 1e3 );
	// optimizer.AddTerminationChecker( iterationChecker );

	trainProblem.Invalidate();
	testProblem.Invalidate();
	trainProblem.ForepropAll();
	testProblem.ForepropAll();
	trainProblem.losses.ParentCost::Foreprop();
	testProblem.losses.ParentCost::Foreprop();
	std::cout << "initial train avg loss: " << trainProblem.losses.ParentCost::GetOutput() << std::endl;
	std::cout << "initial train max loss: " << trainProblem.losses.ParentCost::ComputeMax() << std::endl;
	std::cout << "initial test avg loss: " << testProblem.losses.ParentCost::GetOutput() << std::endl;
	std::cout << "initial test max loss: " << testProblem.losses.ParentCost::ComputeMax() << std::endl;

	std::cout << "Beginning optimization..." << std::endl;
	optimizer.ResetAll();
	trainProblem.Resample();
	OptimizationResults result = optimizer.Optimize( trainProblem );
	std::cout << "Terminated with condition: " << result.status << std::endl;
	std::cout << "Final net: " << std::endl << reg.net << std::endl;

	trainProblem.Invalidate();
	testProblem.Invalidate();
	trainProblem.ForepropAll();
	testProblem.ForepropAll();
	trainProblem.losses.ParentCost::Foreprop();
	testProblem.losses.ParentCost::Foreprop();
	std::cout << "train avg loss: " << trainProblem.losses.ParentCost::GetOutput() << std::endl;
	std::cout << "train max loss: " << trainProblem.losses.ParentCost::ComputeMax() << std::endl;
	std::cout << "test avg loss: " << testProblem.losses.ParentCost::GetOutput() << std::endl;
	std::cout << "test max loss: " << testProblem.losses.ParentCost::ComputeMax() << std::endl;

	for( unsigned int i = 0; i < numTest; i++ )
	{
		std::cout << "xtest: " << xTest[i] << " ytest: " << yTest[i]
		          << " regout: " << testProblem.regressors[i].GetOutput() << std::endl;
	}

	return 0;
}