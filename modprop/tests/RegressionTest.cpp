#include "modprop/compo/ConstantRegressor.hpp"
#include "modprop/compo/ExponentialWrapper.hpp"
#include "modprop/compo/OffsetWrapper.hpp"
#include "modprop/compo/ModifiedCholeskyWrapper.hpp"
#include "modprop/compo/TransformWrapper.hpp"

#include "modprop/utils/LowerTriangular.hpp"
#include "modprop/utils/Randomization.hpp"

#include "modprop/optim/GaussianLogLikelihoodCost.hpp"

#include "modprop/neural/NetworkTypes.h"

#include <deque>
#include <ctime>
#include <iostream>

using namespace percepto;

typedef ExponentialWrapper ExpModule;
typedef ModifiedCholeskyWrapper PSDModule;
typedef OffsetWrapper<MatrixType> OffsetModule;

typedef TransformWrapper TransEst;
typedef GaussianLogLikelihoodCost GLL;

double ClocksToMicrosecs( clock_t c )
{
	return c * 1E6 / CLOCKS_PER_SEC;
}

unsigned int matDim = 3;
unsigned int dFeatDim = 5;

unsigned int lOutDim = matDim*(matDim-1)/2;
unsigned int dOutDim = matDim;

unsigned int numHiddenLayers = 1;
unsigned int layerWidth = 10;

struct Regressor
{
	TerminalSource<VectorType> dInput;
	ConstantVectorRegressor lReg;
	ReLUNet dReg;
	ExpModule expReg;
	PSDModule psdReg;
	OffsetModule pdReg;
	TransEst transReg;
	GLL gll;

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
		transReg.SetSource( &pdReg );
		gll.SetSource( &transReg );
	}

	Regressor( const Regressor& other )
	: lReg( other.lReg ), dReg( other.dReg ), pdReg( other.pdReg )
	{
		dReg.SetSource( &dInput );
		expReg.SetSource( &dReg.GetOutputSource() );
		psdReg.SetLSource( &lReg );
		psdReg.SetDSource( &expReg );
		pdReg.SetSource( &psdReg );
		transReg.SetSource( &pdReg );
		gll.SetSource( &transReg );
	}

	void Foreprop()
	{
		lReg.Foreprop();
		dInput.Foreprop();
	}

	void Backprop()
	{
		gll.Backprop( MatrixType() );
	}

};

int main( void )
{

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "D Feature dim: " << dFeatDim << std::endl;
	std::cout << "L Output dim: " << lOutDim << std::endl;
	std::cout << "D Output dim: " << dOutDim << std::endl;
	
	Regressor reg;
	Parameters::Ptr lParams = reg.lReg.CreateParameters();
	VectorType temp( lParams->ParamDim() );
	randomize_vector( temp );
	lParams->SetParamsVec( temp );

	Parameters::Ptr dParams = reg.dReg.CreateParameters();
	temp = VectorType( dParams->ParamDim() );
	randomize_vector( temp );
	dParams->SetParamsVec( temp );

	// Test speed of various functions here
	unsigned int popSize = 1000;

	// // 1. Generate test set
	std::deque<Regressor> regressors;
	std::vector<VectorType> inputVals( popSize );
	for( unsigned int i = 0; i < popSize; i++ )
	{
		regressors.emplace_back( reg );

		VectorType dInput = VectorType::Random( dFeatDim );
		MatrixType transform = MatrixType::Random( matDim, matDim );
		VectorType sample = VectorType::Random( matDim );
		
		regressors[i].transReg.SetTransform( transform );
		regressors[i].gll.SetSample( sample );

		inputVals[i] = dInput;
	}

	// 2. Test speed of forward pass
	std::cout << "Testing pipeline forward pass speed..." << std::endl;
	double regOutput, lastOutput;
	clock_t start = clock();
	for( unsigned int i = 0; i < popSize; i++ )
	{
		regressors[i].dInput.SetOutput( inputVals[i] );
		regressors[i].Foreprop();
		regOutput = regressors[i].gll.GetOutput();
		if( i > 1 && regOutput == lastOutput )
		{
			std::cout << "last: " << lastOutput << std::endl;
			std::cout << "new: " << regOutput << std::endl;
			exit( -1 );
		}
		lastOutput = regOutput;
	}
	clock_t finish = clock();
	double totalTime = ClocksToMicrosecs( finish - start );
	std::cout << "Took " << totalTime << " us to forward-pass "
	          << popSize << " outputs at " << totalTime / popSize << " us per regression."
	          << std::endl;

	// 4. Test gradient calculation speed
	std::cout << "Testing Gaussian LL gradient calculation speed..." << std::endl;
	for( unsigned int i = 0; i < popSize; i++ )
	{
		regressors[i].Backprop();
	}
	finish = clock();
	totalTime = ClocksToMicrosecs( finish - start );
	std::cout << "Took " << totalTime << " us to evaluate "
	          << popSize << " derivatives at " << totalTime / popSize << " us per sample."
	          << std::endl;
}


