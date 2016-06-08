#include "percepto/compo/ConstantRegressor.hpp"
#include "percepto/compo/ExponentialWrapper.hpp"
#include "percepto/compo/OffsetWrapper.hpp"
#include "percepto/compo/ModifiedCholeskyWrapper.hpp"

#include "percepto/compo/InputWrapper.hpp"
#include "percepto/compo/TransformWrapper.hpp"

#include "percepto/utils/LowerTriangular.hpp"
#include "percepto/utils/Randomization.hpp"

#include "percepto/optim/GaussianLogLikelihoodCost.hpp"

#include "percepto/neural/NetworkTypes.h"


#include <ctime>
#include <iostream>

using namespace percepto;

typedef InputWrapper<ReLUNet> BaseModule;
typedef ExponentialWrapper<BaseModule> ExpModule;
typedef ModifiedCholeskyWrapper<ConstantRegressor, ExpModule> PSDModule;
typedef OffsetWrapper<PSDModule> PDModule;
typedef InputChainWrapper<BaseModule,PDModule> CovEstimator;

typedef InputWrapper<CovEstimator> CovEstimate;
typedef TransformWrapper<CovEstimate> TransEst;
typedef GaussianLogLikelihoodCost<CovEstimate> GLL;

double ClocksToMicrosecs( clock_t c )
{
	return c * 1E6 / CLOCKS_PER_SEC;
}

int main( void )
{

	unsigned int matDim = 6;

	TriangularMapping triMap( matDim - 1 );

	unsigned int dFeatDim = 5;
	unsigned int lOutDim = triMap.NumPositions();
	unsigned int dOutDim = matDim;

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "D Feature dim: " << dFeatDim << std::endl;
	std::cout << "L Output dim: " << lOutDim << std::endl;
	std::cout << "D Output dim: " << dOutDim << std::endl;

	ConstantRegressor lReg( lOutDim, 1 );

	HingeActivation relu( 1.0, 1E-3 );
	ReLUNet dReg( dFeatDim, dOutDim, 1, 10, relu );
	BaseModule dRegWrapper( dReg );
	ExpModule expReg( dRegWrapper );
	PSDModule psdReg( lReg, expReg );
	PDModule pdReg( psdReg, 1E-3 * MatrixType::Identity( matDim, matDim ) );
	CovEstimator pdEst( dRegWrapper, pdReg );

	ParametricWrapper parametrics;
	parametrics.AddParametric( &lReg );
	parametrics.AddParametric( &dReg );

	// Test speed of various functions here
	unsigned int popSize = 1000;

	// 1. Generate test set
	std::vector<CovEstimate> estimates;
	std::vector<TransEst> transforms;
	std::vector<GLL> testCosts;

	estimates.reserve( popSize );
	transforms.reserve( popSize );
	testCosts.reserve( popSize );
	for( unsigned int i = 0; i < popSize; i++ )
	{
		VectorType dInput = VectorType::Random( dFeatDim );

		MatrixType transform = MatrixType::Random( matDim, matDim );
		VectorType sample = VectorType::Random( matDim );

		estimates.emplace_back( pdEst, dInput );
		transforms.emplace_back( estimates.back(), transform );
		testCosts.emplace_back( estimates.back(), sample );
	}

	// 2. Test speed of matrix regression
	std::cout << "Testing LDL regression speed..." << std::endl;
	MatrixType regOutput;
	clock_t start = clock();
	for( unsigned int i = 0; i < popSize; i++ )
	{
		regOutput = estimates[i].Evaluate();
	}
	clock_t finish = clock();
	double totalTime = ClocksToMicrosecs( finish - start );
	std::cout << "Took " << totalTime << " us to regress "
	          << popSize << " outputs at " << totalTime / popSize << " us per regression."
	          << std::endl;

	// 3. Test speed of Gaussian cost calculation
	std::cout << "Testing Gaussian LL cost speed..." << std::endl;
	GLL::OutputType llOutput;
	start = clock();
	for( unsigned int i = 0; i < popSize; i++ )
	{
		llOutput = testCosts[i].Evaluate();
	}
	finish = clock();
	totalTime = ClocksToMicrosecs( finish - start );
	std::cout << "Took " << totalTime << " us to evaluate "
	          << popSize << " outputs at " << totalTime / popSize << " us per sample."
	          << std::endl;

	// 4. Test gradient calculation speed
	MatrixType dodx = MatrixType::Identity(1,1);
	std::cout << "Testing Gaussian LL gradient calculation speed with "
	          << parametrics.ParamDim() << " parameters. " << std::endl;
	for( unsigned int i = 0; i < popSize; i++ )
	{
		//llOutput = testCosts[i].Derivative( ind );
		testCosts[i].Backprop( dodx );
	}
	finish = clock();
	totalTime = ClocksToMicrosecs( finish - start );
	std::cout << "Took " << totalTime << " us to evaluate "
	          << popSize << " derivatives at " << totalTime / popSize << " us per sample."
	          << std::endl;
}


