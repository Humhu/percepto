#include "percepto/LinearRegressor.hpp"
#include "percepto/ExponentialWrapper.hpp"
#include "percepto/AffineWrapper.hpp"

#include "percepto/pdreg/ModifiedCholeskyWrapper.hpp"
#include "percepto/pdreg/LowerTriangular.hpp"

#include "percepto/GaussianLogLikelihoodCost.hpp"

#include <ctime>
#include <iostream>

using namespace percepto;

typedef ExponentialWrapper<LinearRegressor> ExpRegressor;
typedef ModifiedCholeskyWrapper<LinearRegressor,ExpRegressor> ModCholRegressor;

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

	LinearRegressor linreg( LinearRegressor::ParamType::Random( lOutDim, lFeatDim ) );

	ExpRegressor expreg( ExpRegressor::ParamType::Random( dOutDim, dFeatDim ) );

	ModCholRegressor mcReg( linreg, expreg, 
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


