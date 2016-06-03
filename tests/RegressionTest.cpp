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

// TODO Implement ConstantRegressor
typedef ExponentialWrapper<ReLUNet> ExpRegressor;
typedef ModifiedCholeskyWrapper<ConstantRegressor, ExpRegressor> PSDReg;
typedef OffsetWrapper<PSDReg> PDReg;

typedef InputWrapper<PDReg> CovEstimate;
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

	unsigned int lFeatDim = 5;
	unsigned int dFeatDim = 5;
	unsigned int lOutDim = triMap.NumPositions();
	unsigned int dOutDim = matDim;

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "L Feature dim: " << lFeatDim << std::endl;
	std::cout << "D Feature dim: " << dFeatDim << std::endl;
	std::cout << "L Output dim: " << lOutDim << std::endl;
	std::cout << "D Output dim: " << dOutDim << std::endl;

	ConstantRegressor lReg( VectorType::Random( lOutDim ) );

	HingeActivation relu( 1.0, 1E-3 );
	ReLUNet dReg( dFeatDim, dOutDim, 1, 10, relu );
	VectorType params = dReg.GetParamsVec();
	randomize_vector( params );
	dReg.SetParamsVec( params );

	ExpRegressor expReg( dReg );

	PSDReg psdReg( lReg, expReg );
	PDReg pdReg( psdReg, 1E-3 * MatrixType::Identity( matDim, matDim ) );

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
		PDReg::InputType input;
		input.lInput = VectorType::Random( lFeatDim );
		input.dInput = VectorType::Random( dFeatDim );

		MatrixType transform = MatrixType::Random( matDim, matDim );
		VectorType sample = VectorType::Random( matDim );

		estimates.emplace_back( pdReg, input );
		transforms.emplace_back( estimates.back(), transform );
		testCosts.emplace_back( estimates.back(), sample );
	}

	// 2. Test speed of matrix regression
	std::cout << "Testing LDL regression speed..." << std::endl;
	MatrixType regOutput;
	clock_t start = clock();
	for( unsigned int i = 0; i < popSize; i++ )
	{
		regOutput = pdReg.Evaluate( estimates[i].input );
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
	MatrixType grad;
	std::cout << "Testing Gaussian LL gradient calculation speed with "
	          << testCosts[0].ParamDim() << " parameters. " << std::endl;
	for( unsigned int i = 0; i < popSize; i++ )
	{
		for( unsigned int ind = 0; ind < testCosts[0].ParamDim(); ind++ )
		{
			//llOutput = testCosts[i].Derivative( ind );
			grad = BackpropGradient( testCosts[i] );
		}
	}
	finish = clock();
	totalTime = ClocksToMicrosecs( finish - start );
	std::cout << "Took " << totalTime << " us to evaluate "
	          << popSize << " derivatives at " << totalTime / popSize << " us per sample."
	          << std::endl;
}


