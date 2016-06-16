#include "percepto/compo/AdditiveWrapper.hpp"
#include "percepto/compo/ConstantRegressor.hpp"
#include "percepto/compo/ExponentialWrapper.hpp"
#include "percepto/compo/OffsetWrapper.hpp"
#include "percepto/compo/ModifiedCholeskyWrapper.hpp"
#include "percepto/compo/TransformWrapper.hpp"
#include "percepto/compo/InverseWrapper.hpp"

#include "percepto/utils/LowerTriangular.hpp"
#include "percepto/utils/Randomization.hpp"
#include "percepto/utils/Derivatives.hpp"

#include "percepto/optim/GaussianLogLikelihoodCost.hpp"
#include "percepto/optim/ParameterL2Cost.hpp"

#include "percepto/neural/NetworkTypes.h"

#include <ctime>
#include <iostream>
#include <deque>

using namespace percepto;

double ClocksToMicrosecs( clock_t c )
{
	return c * 1E6 / CLOCKS_PER_SEC;
}

unsigned int matDim = 6;
unsigned int dFeatDim = 5;

unsigned int lOutDim = matDim*(matDim-1)/2;
unsigned int dOutDim = matDim;

unsigned int numHiddenLayers = 1;
unsigned int layerWidth = 10;

struct Regressor
{
	// TerminalSource<VectorType> dInput;
	ConstantVectorRegressor lReg;
	ConstantVectorRegressor dReg;
	// ReLUNet dReg;
	ExponentialWrapper<VectorType> expReg;
	ModifiedCholeskyWrapper psdReg;
	OffsetWrapper<MatrixType> pdReg;
	TransformWrapper transReg;
	InverseWrapper<EigLDL> invReg;
	GaussianLogLikelihoodCost gll;

	Regressor()
	: lReg( lOutDim ), 
	// dReg( dFeatDim, dOutDim, numHiddenLayers, layerWidth,
	      // HingeActivation( 1.0, 1E-3 ) )
	  dReg( matDim )
	{
		// dReg.SetSource( &dInput );
		// expReg.SetSource( &dReg.GetOutputSource() );
		expReg.SetSource( &dReg );
		psdReg.SetLSource( &lReg );
		psdReg.SetDSource( &expReg );
		pdReg.SetSource( &psdReg );
		pdReg.SetOffset( 1E-9 * MatrixType::Identity( matDim, matDim ) );
		transReg.SetSource( &pdReg );
		invReg.SetSource( &transReg );
		gll.SetSource( &invReg );
	}

	Regressor( const Regressor& other )
	: lReg( other.lReg ), dReg( other.dReg )
	{
		expReg.SetSource( &dReg );
		psdReg.SetLSource( &lReg );
		psdReg.SetDSource( &expReg );
		pdReg.SetSource( &psdReg );
		pdReg.SetOffset( 1E-9 * MatrixType::Identity( matDim, matDim ) );
		transReg.SetSource( &pdReg );
		invReg.SetSource( &transReg );
		gll.SetSource( &invReg );
	}

	void Invalidate()
	{
		lReg.Invalidate();
		dReg.Invalidate();
		// dInput.Invalidate();
	}

	void Foreprop()
	{
		lReg.Foreprop();
		// dInput.Foreprop();
		dReg.Foreprop();
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
	// VectorType temp = lParams->GetParamsVec();
	// randomize_vector( temp );
	// lParams->SetParamsVec( temp );
	lParams->SetParamsVec( VectorType::Zero( lParams->ParamDim() ) );

	Parameters::Ptr dParams = reg.dReg.CreateParameters();
	dParams->SetParamsVec( VectorType::Zero( dParams->ParamDim() ) );
	// std::vector<Parameters::Ptr> dParams = reg.dReg.CreateParameters();
	// for( unsigned int i = 0; i < dParams.size(); i++ )
	// {
	// 	temp = dParams[i]->GetParamsVec();
	// 	randomize_vector( temp );
	// 	dParams[i]->SetParamsVec( temp );
	// }

	ParameterWrapper paramWrapper;
	paramWrapper.AddParameters( lParams );
	paramWrapper.AddParameters( dParams );
	// allParams.insert( allParams.end(), dParams.begin(), dParams.end() );

	// Test derivatives of various functions here
	unsigned int popSize = 100;

	// // 1. Generate test set
	std::deque<Regressor> regressors; //( popSize );
	std::deque<VectorType> inputVals; //( popSize );
	for( unsigned int i = 0; i < popSize; i++ )
	{
		VectorType dInput = VectorType::Random( dFeatDim );
		MatrixType transform = MatrixType::Random( matDim-1, matDim );
		VectorType sample = VectorType::Random( matDim-1 );
		
		regressors.emplace_back( reg );
		inputVals.emplace_back( dInput );
		// regressors[i].lReg.SetParameters( lParams );
		// regressors[i].dReg.SetParameters( dParams );
		regressors[i].transReg.SetTransform( transform );
		regressors[i].gll.SetSample( sample );
		// regressors[i].dInput.SetOutput( dInput );
		// inputVals[i] = dInput;
	}

	std::vector<double>::iterator iter;

	double maxSeen, acc;

	// 3. Test ANN gradients
	// std::cout << "Testing ANN gradients..." << std::endl;
	// maxSeen = -std::numeric_limits<double>::infinity();
	// acc = 0;
	// for( unsigned int i = 0; i < popSize; i++ )
	// {
	// 	std::vector<double> errors = EvalMatDeriv( regressors[i], 
	// 	                                           // regressors[i].dReg.GetOutputSource(),
	// 	                                           regressors[i].dReg,
	// 	                                           paramWrapper );
	// 	iter = std::max_element( errors.begin(), errors.end() );
	// 	acc += *iter;
	// 	if( *iter > maxSeen ) { maxSeen = *iter; }
	// }
	// std::cout << "Max error: " << maxSeen << std::endl;
	// std::cout << "Avg max error: " << acc / popSize << std::endl;

	// 4. Test cholesky gradients
	std::cout << "Testing Modified Cholesky gradients..." << std::endl;
	maxSeen = -std::numeric_limits<double>::infinity();
	acc = 0;
	for( unsigned int i = 0; i < popSize; i++ )
	{
		std::vector<double> errors = EvalMatDeriv( regressors[i], 
		                                           regressors[i].pdReg,
		                                           paramWrapper );
		iter = std::max_element( errors.begin(), errors.end() );
		acc += *iter;
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;
	std::cout << "Avg max error: " << acc / popSize << std::endl;
	
	// 5. Test transformed and damped cholesky gradients
	std::cout << "Testing transformed gradients..." << std::endl;
	maxSeen = -std::numeric_limits<double>::infinity();
	acc = 0;
	for( unsigned int i = 0; i < popSize; i++ )
	{
		std::vector<double> errors = EvalMatDeriv( regressors[i],
		                                           regressors[i].transReg,
		                                           paramWrapper );
		iter = std::max_element( errors.begin(), errors.end() );
		acc += *iter;
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;
	std::cout << "Avg max error: " << acc / popSize << std::endl;

	std::cout << "Testing inverse gradients..." << std::endl;
	maxSeen = -std::numeric_limits<double>::infinity();
	acc = 0;
	for( unsigned int i = 0; i < popSize; i++ )
	{
		std::vector<double> errors = EvalMatDeriv( regressors[i],
		                                           regressors[i].invReg,
		                                           paramWrapper );
		iter = std::max_element( errors.begin(), errors.end() );
		acc += *iter;
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;
	std::cout << "Avg max error: " << acc / popSize << std::endl;

	// 6. Test log-likelihood gradients
	std::cout << "Testing log-likelihood gradients..." << std::endl;
	maxSeen = -std::numeric_limits<double>::infinity();
	acc = 0;
	for( unsigned int i = 0; i < popSize; i++ )
	{
		std::vector<double> errors = EvalCostDeriv( regressors[i], 
		                                            regressors[i].gll,
		                                            paramWrapper );
		iter = std::max_element( errors.begin(), errors.end() );
		acc += *iter;
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;
	std::cout << "Avg max error: " << acc / popSize << std::endl;

	// 7. Test penalized log-likelihood gradients
	// std::cout << "Testing penalized log-likelihood gradients..." << std::endl;
	// maxSeen = -std::numeric_limits<double>::infinity();
	// acc = 0;
	// ParameterL2Cost l2cost;
	// l2cost.SetParameters( &paramWrapper );
	// l2cost.SetWeight( 1E-3 );
	// for( unsigned int i = 0; i < popSize; i++ )
	// {
	// 	AdditiveWrapper<double> penalizedCost;
	// 	penalizedCost.SetSourceA( &regressors[i].gll );
	// 	penalizedCost.SetSourceB( &l2cost );
	// 	std::vector<double> errors = EvalCostDeriv( regressors[i],
	// 	                                            penalizedCost, 
	// 	                                            paramWrapper );
	// 	iter = std::max_element( errors.begin(), errors.end() );
	// 	acc += *iter;
	// 	if( *iter > maxSeen ) { maxSeen = *iter; }
	// }
	// std::cout << "Max error: " << maxSeen << std::endl;
	// std::cout << "Avg max error: " << acc / popSize << std::endl;
	
}	


