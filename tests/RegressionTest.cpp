#include "percepto/compo/LinearRegressor.hpp"
#include "percepto/compo/ExponentialWrapper.hpp"
#include "percepto/compo/AffineWrapper.hpp"
#include "percepto/compo/ModifiedCholeskyWrapper.hpp"

#include "percepto/utils/LowerTriangular.hpp"
#include "percepto/utils/Randomization.hpp"

#include "percepto/optim/GaussianLogLikelihoodCost.hpp"

#include "percepto/neural/NetworkTypes.h"


#include <ctime>
#include <iostream>

using namespace percepto;

// TODO Implement ConstantRegressor
typedef ReLUNet BaseRegressor;
typedef ExponentialWrapper<BaseRegressor> ExpRegressor;
typedef ModifiedCholeskyWrapper<BaseRegressor, ExpRegressor> ModCholRegressor;

typedef AffineWrapper<ModCholRegressor> AffineModCholRegressor;

typedef GaussianLogLikelihoodCost<AffineModCholRegressor> GLL;

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

	//BaseRegressor lReg( BaseRegressor::ParamType::Random( lOutDim, lFeatDim ) );
	HingeActivation relu( 1.0, 1E-3 );
	BaseRegressor lReg = BaseRegressor::create_zeros( lFeatDim, lOutDim, 1, 10, relu );
	VectorType params = lReg.GetParamsVec();
	randomize_vector( params );
	lReg.SetParamsVec( params );

	//ExpRegressor expreg( ExpRegressor::ParamType::Random( dOutDim, dFeatDim ) );
	BaseRegressor dReg = BaseRegressor::create_zeros( dFeatDim, dOutDim, 1, 10, relu );
	params = dReg.GetParamsVec();
	randomize_vector( params );
	dReg.SetParamsVec( params );

	ExpRegressor expreg( dReg );

	ModCholRegressor mcReg( lReg, expreg, 
	                        1E-3 * MatrixType::Identity( matDim, matDim ) );

	AffineModCholRegressor amcReg( mcReg );

	// Test speed of various functions here
	unsigned int popSize = 1000;

	// 1. Generate test set
	std::vector<GLL> testCosts;
	testCosts.reserve( popSize );
	for( unsigned int i = 0; i < popSize; i++ )
	{
		typedef ModCholRegressor::LRegressorType::InputType LInputType;
		typedef ModCholRegressor::DRegressorType::InputType DInputType;
		
		ModCholRegressor::InputType mcInput;
		mcInput.lInput = LInputType::Random( lFeatDim );
		mcInput.dInput = DInputType::Random( dFeatDim );

		AffineModCholRegressor::InputType amcInput;
		amcInput.baseInput = mcInput;
		amcInput.transform = MatrixType::Random( matDim, matDim );
		amcInput.offset = MatrixType::Identity( matDim, matDim );

		VectorType sample = VectorType::Random( matDim );
		testCosts.emplace_back( amcInput, sample, amcReg );
	}

	// 2. Test speed of matrix regression
	std::cout << "Testing LDL regression speed..." << std::endl;
	AffineModCholRegressor::OutputType regOutput;
	clock_t start = clock();
	for( unsigned int i = 0; i < popSize; i++ )
	{
		regOutput = amcReg.Evaluate( testCosts[i]._input );
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


