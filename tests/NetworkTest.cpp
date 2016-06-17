#include "percepto/PerceptoTypes.h"

#include "percepto/compo/AdditiveWrapper.hpp"
#include "percepto/neural/LinearLayer.hpp"
#include "percepto/neural/FullyConnectedNet.hpp"

#include "percepto/neural/HingeActivation.hpp"
#include "percepto/neural/SigmoidActivation.hpp"
#include "percepto/neural/NullActivation.hpp"
#include "percepto/neural/NetworkTypes.h"

#include "percepto/optim/SquaredLoss.hpp"
#include "percepto/optim/StochasticMeanCost.hpp"
#include "percepto/optim/ParameterL2Cost.hpp"

#include "percepto/optim/OptimizerTypes.h"

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "percepto/utils/Derivatives.hpp"
#include "percepto/utils/Randomization.hpp"

#include <deque>
#include <cstdlib>
#include <iostream>

using namespace percepto;

// Comment the above and uncomment below to use Rectified Linear Units instead
// typedef PerceptronNet TestNet;
typedef ReLUNet TestNet;

unsigned int inputDim = 1;
unsigned int outputDim = 1;
unsigned int numHiddenLayers = 3;
unsigned int layerWidth = 50;
unsigned int batchSize = 10;

struct Regressor
{
	TerminalSource<VectorType> netInput;
	TestNet net;
	SquaredLoss<VectorType> loss;

	Regressor()
	: net( inputDim, outputDim, numHiddenLayers, layerWidth,
	       HingeActivation( 1, 1E-3 ), ReLUNet::OUTPUT_UNRECTIFIED )
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

struct OptimizationProblem
{
	std::deque<Regressor> regressors;
	StochasticMeanCost<double> losses;
	ParameterL2Cost regularizer;
	AdditiveWrapper<double> objective;

	OptimizationProblem() 
	{
		objective.SetSourceA( &losses );
		objective.SetSourceB( &regularizer );
	}

	void Invalidate()
	{
		for( unsigned int i = 0; i < regressors.size(); i++ )
		{
			regressors[i].Invalidate();
		}
		regularizer.Invalidate();
	}

	void Foreprop()
	{
		for( unsigned int i = 0; i < regressors.size(); i++ )
		{
			regressors[i].Foreprop();
		}
		regularizer.Foreprop();
	}

	double GetOutput()
	{
		return objective.GetOutput();
	}

	void Backprop()
	{
		objective.Backprop( MatrixType() );
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

	// Perceptron initialization
	// SigmoidActivation act;
	// PerceptronSubnet subnet( inputDim, layerWidth, numHiddenLayers, 
	//                          layerWidth, act );
	// LinearLayer<NullActivation> outputLayer( layerWidth, outputDim, NullActivation() );
	// InputWrapper<PerceptronSubnet> subnetWrapper( subnet );
	// PerceptronSeries netSeries( subnetWrapper, outputLayer );
	// PerceptronNet testNet( subnetWrapper, netSeries );

	// netParametrics.AddParametric( &subnet );
	// netParametrics.AddParametric( &outputLayer );

	// // ReLU initialization
	Regressor reg;
	Parameters::Ptr params = reg.net.CreateParameters();

	// Randomize parameters
	VectorType p( params->ParamDim() );
	randomize_vector( p );
	params->SetParamsVec( p );

	// Create the loss functions
	std::cout << "Generating losses..." << std::endl;
	OptimizationProblem trainProblem;
	OptimizationProblem testProblem;
	
	// NOTE If we don't reserve, the vector resizing and moving may
	// invalidate the references. Alternatively we can use a deque
	trainProblem.losses.SetBatchSize( batchSize );
	trainProblem.regularizer.SetParameters( params.get() );
	trainProblem.regularizer.SetWeight( l2Weight );
	for( unsigned int i = 0; i < numTrain; i++ )
	{
		trainProblem.regressors.emplace_back( reg );
		trainProblem.regressors[i].netInput.SetOutput( xTrain[i] );
		trainProblem.regressors[i].loss.SetTarget( yTrain[i] );
		trainProblem.losses.AddSource( &trainProblem.regressors[i].loss );
	}

	testProblem.losses.SetBatchSize( batchSize );
	testProblem.regularizer.SetParameters( params.get() );
	testProblem.regularizer.SetWeight( l2Weight );
	for( unsigned int i = 0; i < numTest; i++ )
	{
		testProblem.regressors.emplace_back( reg );
		testProblem.regressors[i].netInput.SetOutput( xTest[i] );
		testProblem.regressors[i].loss.SetTarget( yTest[i] );
		testProblem.losses.AddSource( &testProblem.regressors[i].loss );
	}

	AdamStepper stepper;
	
	SimpleConvergenceCriteria criteria;
	criteria.maxRuntime = 20;
	criteria.minElementGradient = 1E-3;
	//criteria.minObjectiveDelta = 1E-3;
	SimpleConvergence convergence( criteria );

	// std::cout << "Evaluating derivatives..." << std::endl;
	// double acc = 0;
	// unsigned int num = 0;
	// double maxSeen = -std::numeric_limits<double>::infinity();
	// for( unsigned int i = 0; i < numTrain; i++ )
	// {
	// 	std::vector<double> errors = EvalCostDeriv( testLosses[i], params );
	// 	double trialMax = *std::max_element( errors.begin(), errors.end() );
	// 	if( trialMax > maxSeen ) { maxSeen = trialMax; }
	// 	acc += trialMax;
	// 	++num;
	// }
	// std::cout << "Mean max error: " << acc / num << std::endl;
	// std::cout << "Max overall error: " << maxSeen << std::endl;

	trainProblem.Invalidate();
	testProblem.Invalidate();
	trainProblem.Foreprop();
	testProblem.Foreprop();
	trainProblem.losses.ParentCost::Foreprop();
	testProblem.losses.ParentCost::Foreprop();
	std::cout << "initial train avg loss: " << trainProblem.losses.ParentCost::GetOutput() << std::endl;
	std::cout << "initial train max loss: " << trainProblem.losses.ParentCost::ComputeMax() << std::endl;
	std::cout << "initial test avg loss: " << testProblem.losses.ParentCost::GetOutput() << std::endl;
	std::cout << "initial test max loss: " << testProblem.losses.ParentCost::ComputeMax() << std::endl;

	std::cout << "Beginning optimization..." << std::endl;
	AdamOptimizer optimizer( stepper, convergence, *params );
	optimizer.Optimize( trainProblem );

	trainProblem.Invalidate();
	testProblem.Invalidate();
	trainProblem.Foreprop();
	testProblem.Foreprop();
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