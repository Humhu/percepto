#include "percepto/PerceptoTypes.h"

#include "percepto/neural/LinearLayer.hpp"
#include "percepto/neural/FullyConnectedNet.hpp"
#include "percepto/neural/NetWrapper.hpp"

#include "percepto/neural/HingeActivation.hpp"
#include "percepto/neural/SigmoidActivation.hpp"
#include "percepto/neural/NullActivation.hpp"

#include "percepto/SquaredLoss.hpp"
#include "percepto/StochasticPopulationLoss.hpp"
#include "percepto/L2ParameterLoss.hpp"

#include "percepto/optim/Optimizer.hpp"
#include "percepto/optim/AdamStepper.hpp"
#include "percepto/optim/SimpleConvergence.hpp"

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <cstdlib>
#include <iostream>

using namespace percepto;

typedef FullyConnectedNet<HingeActivation> ReLUNet;
typedef FullyConnectedNet<SigmoidActivation> PerceptronNet;
typedef LinearLayer<NullActivation> UnrectifiedLinearLayer;
typedef NetWrapper<PerceptronNet, UnrectifiedLinearLayer> PerceptronOutputNet;

typedef PerceptronOutputNet TestNet;
// Comment the above and unncomment this line to use Rectified Linear Units instead
// typedef ReLUNet TestNet;

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

// For initializing vectors to random in a range
template <typename Derived>
void randomize_vector( Eigen::DenseBase<Derived>& mat, 
                       double minRange = -1.0, double maxRange = 1.0 )
{
	boost::random::mt19937 generator;
	boost::random::random_device rng;
	generator.seed( rng );
	boost::random::uniform_real_distribution<> xDist( minRange, maxRange );

	for( unsigned int i = 0; i < mat.rows(); ++i )
	{
		for( unsigned int j = 0; j < mat.cols(); ++j )
		{
			mat(i,j) = xDist( generator );
		}
	}
}

int main( int argc, char** argv )
{

	unsigned int numTrain = 150;
	unsigned int numTest = 200;
	unsigned int batchSize = 25;
	double l2Weight = 0;

	std::vector<VectorType> xTest, yTest, xTrain, yTrain;
	generate_data( xTest, yTest, numTest );
	generate_data( xTrain, yTrain, numTrain );

	unsigned int inputDim = 1;
	unsigned int outputDim = 1;
	unsigned int numHiddenLayers = 1;
	unsigned int layerWidth = 50;

	// Randomize the net parameters
	std::cout << "Initializing net..." << std::endl;
	std::cout << "Creating linear layers..." << std::endl;

	// Perceptron initialization
	SigmoidActivation act;
	PerceptronNet testHeadNet = PerceptronNet::create_zeros( inputDim, 
	                                                         layerWidth, 
	                                                         numHiddenLayers, 
	                                                         layerWidth, 
	                                                         act );

	// // ReLU initialization
	// HingeActivation act;
	// ReLUNet testHeadNet = ReLUNet::create_zeros( inputDim,
	//                                              layerWidth,
	//                                              numHiddenLayers,
	//                                              layerWidth,
	//                                              act );

	VectorType headParams = testHeadNet.GetParamsVec();
	randomize_vector( headParams );
	testHeadNet.SetParamsVec( headParams );

	std::cout << "Creating output layer..." << std::endl;
	UnrectifiedLinearLayer testOutNet = 
	    UnrectifiedLinearLayer::create_zeros( layerWidth,
	                                          outputDim,
	                                          NullActivation() );
	VectorType outParams = testOutNet.GetParamsVec();
	randomize_vector( outParams );
	testOutNet.SetParamsVec( outParams );

	TestNet testNet( testHeadNet, testOutNet );

	// Create the loss functions
	typedef SquaredLoss<TestNet> Loss;
	typedef StochasticPopulationLoss<Loss> StochasticLoss;
	typedef L2ParameterLoss<StochasticLoss> RegularizedStochasticLoss;

	std::cout << "Generating losses..." << std::endl;
	std::vector<Loss> trainLosses, testLosses;
	for( unsigned int i = 0; i < numTrain; i++ )
	{
		trainLosses.emplace_back( xTrain[i], yTrain[i], testNet );
	}
	StochasticLoss trainLoss( trainLosses, batchSize );
	RegularizedStochasticLoss trainObjective( trainLoss, l2Weight );

	for( unsigned int i = 0; i < numTest; i++ )
	{
		testLosses.emplace_back( xTest[i], yTest[i], testNet );
	}
	StochasticLoss testLoss( testLosses, batchSize );

	AdamStepper stepper;
	
	SimpleConvergenceCriteria criteria;
	criteria.maxRuntime = 180;
	criteria.minElementGradient = 1E-3;
	criteria.minObjectiveDelta = 1E-3;
	SimpleConvergence convergence( criteria );

	std::cout << "Beginning optimization..." << std::endl;
	typedef Optimizer<RegularizedStochasticLoss, AdamStepper, SimpleConvergence> Opt;
	Opt optimizer( trainObjective, stepper, convergence );
	optimizer.Run();

	std::cout << "train avg loss: " << trainLoss.ParentCost::Evaluate() << std::endl;
	std::cout << "train max loss: " << trainLoss.ParentCost::EvaluateMax() << std::endl;
	std::cout << "test avg loss: " << testLoss.ParentCost::Evaluate() << std::endl;
	std::cout << "test max loss: " << testLoss.ParentCost::EvaluateMax() << std::endl;

	// Test the evaluation speed
	unsigned int numEvaluationPoints = 1000;
	unsigned int numTrials = 10;
	std::vector<VectorType> xTiming, yTiming;
	generate_data( xTiming, yTiming, numEvaluationPoints );

	for( unsigned int trial = 0; trial < numTrials; trial++ )
	{
		clock_t startTime = clock();
		for( unsigned int i = 0; i < numEvaluationPoints; i++ )
		{
			testNet.Evaluate( xTiming[i] );
		}
		clock_t endTime = clock();
		double runtime = ((double) endTime - startTime ) / CLOCKS_PER_SEC;
		std::cout << "Time per evaluation: " << runtime/numEvaluationPoints << std::endl;
	}


	return 0;
}