#include "percepto/PerceptoTypes.h"

#include "percepto/compo/AdditiveWrapper.hpp"
#include "percepto/neural/LinearLayer.hpp"
#include "percepto/neural/FullyConnectedNet.hpp"
#include "percepto/compo/SeriesWrapper.hpp"
#include "percepto/compo/InputWrapper.hpp"

#include "percepto/neural/HingeActivation.hpp"
#include "percepto/neural/SigmoidActivation.hpp"
#include "percepto/neural/NullActivation.hpp"
#include "percepto/neural/NetworkTypes.h"

#include "percepto/optim/SquaredLoss.hpp"
#include "percepto/optim/StochasticPopulationCost.hpp"
#include "percepto/optim/ParameterL2Cost.hpp"

#include "percepto/optim/OptimizerTypes.h"

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "percepto/utils/Derivatives.hpp"
#include "percepto/utils/Randomization.hpp"

#include <cstdlib>
#include <iostream>

using namespace percepto;

// Comment the above and uncomment below to use Rectified Linear Units instead
// typedef PerceptronNet TestNet;
typedef ReLUNet TestNet;

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
	unsigned int batchSize = 10;
	double l2Weight = 0;

	std::vector<VectorType> xTest, yTest, xTrain, yTrain;
	generate_data( xTest, yTest, numTest );
	generate_data( xTrain, yTrain, numTrain );

	unsigned int inputDim = 1;
	unsigned int outputDim = 1;
	unsigned int numHiddenLayers = 3;
	unsigned int layerWidth = 50;

	// Randomize the net parameters
	std::cout << "Initializing net..." << std::endl;
	std::cout << "Creating linear layers..." << std::endl;

	ParametricWrapper netParametrics;

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
	HingeActivation act( 1, 1E-1 );
	TestNet testNet( inputDim, outputDim, numHiddenLayers,
	                 layerWidth, act );
	netParametrics.AddParametric( &testNet );

	// Randomize parameters
	VectorType netParams = netParametrics.GetParamsVec();
	randomize_vector( netParams );
	netParametrics.SetParamsVec( netParams );
	ParameterL2Cost l2Loss( netParametrics, l2Weight );

	// Create the loss functions
	typedef InputWrapper<TestNet> NetEst;
	typedef SquaredLoss<NetEst> Loss;
	typedef MeanPopulationCost<Loss> MeanLoss;
	typedef StochasticPopulationCost<Loss> StochasticLoss;
	typedef AdditiveWrapper <StochasticLoss,
	                         ParameterL2Cost> RegularizedStochasticLoss;

	std::cout << "Generating losses..." << std::endl;
	std::vector<NetEst> trainEsts, testEsts;
	std::vector<Loss> trainLosses, testLosses;
	
	// NOTE If we don't reserve, the vector resizing and moving may
	// invalidate the references. Alternatively we can use a deque
	trainEsts.reserve( numTrain );
	trainLosses.reserve( numTrain );
	for( unsigned int i = 0; i < numTrain; i++ )
	{
		trainEsts.emplace_back( testNet, xTrain[i] );
		trainLosses.emplace_back( trainEsts.back(), yTrain[i] );
	}
	StochasticLoss trainLoss( trainLosses, batchSize );
	RegularizedStochasticLoss trainObjective( trainLoss, l2Loss );

	testEsts.reserve( numTest );
	testLosses.reserve( numTest );
	for( unsigned int i = 0; i < numTest; i++ )
	{
		testEsts.emplace_back( testNet, xTest[i] );
		testLosses.emplace_back( testEsts.back(), yTest[i] );
	}
	MeanLoss meanLossS( testLosses );
	StochasticLoss testLoss( testLosses, batchSize );

	AdamStepper stepper;
	
	SimpleConvergenceCriteria criteria;
	criteria.maxRuntime = 600;
	criteria.minElementGradient = 1E-3;
	criteria.minObjectiveDelta = 1E-3;
	SimpleConvergence convergence( criteria );

	// std::cout << "Evaluating derivatives..." << std::endl;
	// double acc = 0;
	// unsigned int num = 0;
	// double maxSeen = -std::numeric_limits<double>::infinity();
	// for( unsigned int i = 0; i < numTrain; i++ )
	// {
	// 	std::vector<double> errors = EvalCostDeriv( testLosses[i], netParametrics );
	// 	double trialMax = *std::max_element( errors.begin(), errors.end() );
	// 	if( trialMax > maxSeen ) { maxSeen = trialMax; }
	// 	acc += trialMax;
	// 	++num;
	// }
	// std::cout << "Mean max error: " << acc / num << std::endl;
	// std::cout << "Max overall error: " << maxSeen << std::endl;

	std::cout << "initial train avg loss: " << trainLoss.ParentCost::Evaluate() << std::endl;
	std::cout << "initial train max loss: " << trainLoss.ParentCost::EvaluateMax() << std::endl;
	std::cout << "initial test avg loss: " << testLoss.ParentCost::Evaluate() << std::endl;
	std::cout << "initial test max loss: " << testLoss.ParentCost::EvaluateMax() << std::endl;

	std::cout << "Beginning optimization..." << std::endl;
	AdamOptimizer optimizer( stepper, convergence, netParametrics );
	optimizer.Optimize( trainObjective );

	std::cout << "train avg loss: " << trainLoss.ParentCost::Evaluate() << std::endl;
	std::cout << "train max loss: " << trainLoss.ParentCost::EvaluateMax() << std::endl;
	std::cout << "test avg loss: " << testLoss.ParentCost::Evaluate() << std::endl;
	std::cout << "test max loss: " << testLoss.ParentCost::EvaluateMax() << std::endl;

	return 0;
}