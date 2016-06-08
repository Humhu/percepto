#include "percepto/compo/AdditiveWrapper.hpp"
#include "percepto/compo/ConstantRegressor.hpp"
#include "percepto/compo/ExponentialWrapper.hpp"
#include "percepto/compo/OffsetWrapper.hpp"
#include "percepto/compo/ModifiedCholeskyWrapper.hpp"

#include "percepto/compo/InputWrapper.hpp"
#include "percepto/compo/TransformWrapper.hpp"

#include "percepto/utils/LowerTriangular.hpp"
#include "percepto/utils/Randomization.hpp"
#include "percepto/utils/Derivatives.hpp"

#include "percepto/optim/GaussianLogLikelihoodCost.hpp"
#include "percepto/optim/ParameterL2Cost.hpp"

#include "percepto/neural/NetworkTypes.h"

#include <ctime>
#include <iostream>

using namespace percepto;

typedef TriangularMapping TriMapd;

typedef ReLUNet BaseRegressor;
typedef InputWrapper<BaseRegressor> BaseModule;
typedef ExponentialWrapper<BaseModule> ExpModule;
typedef ModifiedCholeskyWrapper<ConstantRegressor, ExpModule> PSDModule;
typedef OffsetWrapper<PSDModule> PDModule;
typedef InputChainWrapper<BaseModule,PDModule> CovEstimate;

typedef TransformWrapper<CovEstimate> TransCovEstimate;
typedef GaussianLogLikelihoodCost<TransCovEstimate> GLL;
typedef AdditiveWrapper<GLL,ParameterL2Cost> PenalizedGLL;

int main( void )
{

	unsigned int matDim = 3;

	TriMapd triMap( matDim - 1 );

	unsigned int dFeatDim = 5;
	unsigned int lOutDim = triMap.NumPositions();
	unsigned int dOutDim = matDim;
	
	unsigned int dNumHiddenLayers = 1;
	unsigned int dLayerWidth = 10;

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "D Feature dim: " << dFeatDim << std::endl;
	std::cout << "L Output dim: " << lOutDim << std::endl;
	std::cout << "D Output dim: " << dOutDim << std::endl;

	ConstantRegressor lReg( MatrixType::Random( lOutDim, 1 ) );
	
	HingeActivation relu( 1.0, 1E-3 );
	BaseRegressor dReg ( dFeatDim, dOutDim, dNumHiddenLayers, 
	                     dLayerWidth, relu );
	VectorType params = dReg.GetParamsVec();
	randomize_vector( params );
	dReg.SetParamsVec( params );

	BaseModule baseIn( dReg );
	ExpModule expreg( baseIn );
	PSDModule psdReg( lReg, expreg );
	PDModule pdReg( psdReg, 1E-3 * MatrixType::Identity( matDim, matDim ) );
	ParametricWrapper pdParameters;
	
	pdParameters.AddParametric( &lReg );
	pdParameters.AddParametric( &dReg );

	// Test derivatives of various functions here
	unsigned int popSize = 100;

	// 1. Generate test set
	std::vector<CovEstimate> estimates;
	std::vector<TransCovEstimate> transformedEsts;
	std::vector<GLL> likelihoods;

	estimates.reserve( popSize );
	transformedEsts.reserve( popSize );
	likelihoods.reserve( popSize );
	for( unsigned int i = 0; i < popSize; i++ )
	{
		VectorType dInput( dFeatDim );
		randomize_vector( dInput );

		estimates.emplace_back( baseIn, pdReg, dInput );
		transformedEsts.emplace_back( estimates.back(), MatrixType::Random( matDim - 1, matDim ) );
		likelihoods.emplace_back( transformedEsts.back(), VectorType::Random( matDim - 1 ) );
	}

	std::vector<double>::iterator iter;

	// 4. Test cholesky gradients
	std::cout << "Testing matrix gradients..." << std::endl;
	double maxSeen = -std::numeric_limits<double>::infinity();
	double acc = 0;
	for( unsigned int i = 0; i < popSize; i++ )
	{
		std::vector<double> errors = EvalMatDeriv( estimates[i], pdParameters );
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
		std::vector<double> errors = EvalMatDeriv( transformedEsts[i], pdParameters );
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
		std::vector<double> errors = EvalCostDeriv<GLL>( likelihoods[i], pdParameters );
		iter = std::max_element( errors.begin(), errors.end() );
		acc += *iter;
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;
	std::cout << "Avg max error: " << acc / popSize << std::endl;

	// 7. Test penalized log-likelihood gradients
	std::cout << "Testing penalized log-likelihood gradients..." << std::endl;
	maxSeen = -std::numeric_limits<double>::infinity();
	acc = 0;
	ParameterL2Cost l2cost( pdParameters, 1E-3 );
	for( unsigned int i = 0; i < popSize; i++ )
	{
		PenalizedGLL penalizedCost( likelihoods[i], l2cost );
		std::vector<double> errors = EvalCostDeriv<PenalizedGLL>( penalizedCost, pdParameters );
		iter = std::max_element( errors.begin(), errors.end() );
		acc += *iter;
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;
	std::cout << "Avg max error: " << acc / popSize << std::endl;

}	


