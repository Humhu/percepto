#include "percepto/compo/ConstantRegressor.hpp"
#include "percepto/compo/ExponentialWrapper.hpp"
#include "percepto/compo/OffsetWrapper.hpp"
#include "percepto/compo/ModifiedCholeskyWrapper.hpp"

#include "percepto/compo/InputWrapper.hpp"
#include "percepto/compo/TransformWrapper.hpp"

#include "percepto/utils/LowerTriangular.hpp"
#include "percepto/utils/Randomization.hpp"

#include "percepto/optim/GaussianLogLikelihoodCost.hpp"
#include "percepto/optim/ParameterL2Cost.hpp"

#include "percepto/neural/NetworkTypes.h"

#include <ctime>
#include <iostream>

using namespace percepto;

typedef TriangularMapping TriMapd;

typedef ReLUNet BaseRegressor;
typedef ExponentialWrapper<BaseRegressor> ExpRegressor;
typedef ModifiedCholeskyWrapper<ConstantRegressor, ExpRegressor> PSDRegressor;
typedef OffsetWrapper<PSDRegressor> PDRegressor;

typedef InputWrapper<PDRegressor> CovEstimate;
typedef TransformWrapper<CovEstimate> TransCovEstimate;
typedef GaussianLogLikelihoodCost<TransCovEstimate> GLL;
typedef ParameterL2Cost<GLL> PenalizedGLL;

/*! \brief Tests a gradient calculation numerically by taking a small step and
 * checking the output. */
template <typename Regressor>
std::vector<double> EvalVecDeriv( Regressor& r, 
                                  const typename Regressor::InputType& input )
{
	static double stepSize = 1E-3;

	std::vector<double> errors( r.ParamDim() );

	typedef typename Regressor::OutputType OutType;

	OutType output, testOutput;
	VectorType paramsVec = r.GetParamsVec();
	MatrixType grad = BackpropGradient( r, input );
	output = r.Evaluate( input );

	for( unsigned int ind = 0; ind < r.ParamDim(); ind++ )
	{
		// Take a step and check the output
		VectorType testParamsVec = paramsVec;
		testParamsVec[ind] += stepSize;
		r.SetParamsVec( testParamsVec );
		testOutput = r.Evaluate( input );
		r.SetParamsVec( paramsVec );
		
		double change = ( output - testOutput ).norm();
		double predErr = ( output + grad.row(ind).transpose()*stepSize - testOutput ).norm();
		errors[ind] = predErr/change;
	}
	return errors;
}

template <typename Regressor>
std::vector<double> EvalMatDeriv( Regressor& r )
{
	static double stepSize = 1E-3;

	std::vector<double> errors( r.ParamDim() );

	typedef typename Regressor::OutputType OutType;

	OutType output, testOutput;
	VectorType paramsVec = r.GetParamsVec();
	MatrixType grad = BackpropGradient( r );
	output = r.Evaluate();

	for( unsigned int ind = 0; ind < r.ParamDim(); ind++ )
	{
		// Take a step and check the output
		VectorType testParamsVec = paramsVec;
		testParamsVec[ind] += stepSize;
		r.SetParamsVec( testParamsVec );
		testOutput = r.Evaluate();
		r.SetParamsVec( paramsVec );
		
		VectorType gi = grad.row(ind);
		Eigen::Map<MatrixType> dS( gi.data(), r.OutputSize().rows, r.OutputSize().cols );
		MatrixType predOut = output + dS*stepSize;
		double predErr = ( predOut - testOutput ).norm();

		errors[ind] = predErr;
	}
	return errors;
}

template <typename Cost>
std::vector<double> EvaluateCostDerivative( Cost& cost )
{
	static double stepSize = 1E-3;

	std::vector<double> errors( cost.ParamDim() );

	typedef typename Cost::OutputType OutType;

	OutType output, testOutput;
	VectorType paramsVec = cost.GetParamsVec();
	VectorType grad = BackpropGradient( cost );
	output = cost.Evaluate();

	for( unsigned int ind = 0; ind < cost.ParamDim(); ind++ )
	{
		// Take a step and check the output
		VectorType testParamsVec = paramsVec;
		testParamsVec[ind] += stepSize;
		cost.SetParamsVec( testParamsVec );
		testOutput = cost.Evaluate();
		cost.SetParamsVec( paramsVec );

		double predOut = output + grad(ind)*stepSize;
		double predErr = std::abs( predOut - testOutput );
		errors[ind] = predErr;
	}
	return errors;
}

int main( void )
{

	unsigned int matDim = 3;

	TriMapd triMap( matDim - 1 );

	unsigned int lFeatDim = 1;
	unsigned int dFeatDim = 5;
	unsigned int lOutDim = triMap.NumPositions();
	unsigned int dOutDim = matDim;
	
	unsigned int dNumHiddenLayers = 1;
	unsigned int dLayerWidth = 10;

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "L Feature dim: " << lFeatDim << std::endl;
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

	ExpRegressor expreg( dReg );

	PSDRegressor psdReg( lReg, expreg );
	PDRegressor pdReg( psdReg, 1E-3 * MatrixType::Identity( matDim, matDim ) );

	// Test derivatives of various functions here
	unsigned int popSize = 100;

	// 1. Generate test set
	std::vector<CovEstimate> estimates;
	std::vector<TransCovEstimate> transformedEsts;
	std::vector<GLL> likelihoods;

	transformedEsts.reserve( popSize );
	estimates.reserve( popSize );
	likelihoods.reserve( popSize );
	for( unsigned int i = 0; i < popSize; i++ )
	{
		PDRegressor::InputType pdInput;
		pdInput.lInput = VectorType( lFeatDim );
		pdInput.dInput = VectorType( dFeatDim );
		randomize_vector( pdInput.lInput );
		randomize_vector( pdInput.dInput );

		estimates.emplace_back( pdReg, pdInput );
		transformedEsts.emplace_back( estimates.back(), MatrixType::Random( matDim - 1, matDim ) );
		likelihoods.emplace_back( transformedEsts.back(), VectorType::Random( matDim - 1 ) );
	}

	std::vector<double>::iterator iter;

	// 4. Test cholesky gradients
	std::cout << "Testing regressor gradients..." << std::endl;
	double maxSeen = -std::numeric_limits<double>::infinity();
	double acc = 0;
	for( unsigned int i = 0; i < popSize; i++ )
	{
		std::vector<double> errors = EvalMatDeriv( estimates[i] );
		iter = std::max_element( errors.begin(), errors.end() );
		acc += *iter;
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;
	std::cout << "Avg max error: " << acc / popSize << std::endl;

	// 5. Test transformed and damped cholesky gradients
	std::cout << "Testing affine modified Cholesky gradients..." << std::endl;
	maxSeen = -std::numeric_limits<double>::infinity();
	acc = 0;
	for( unsigned int i = 0; i < popSize; i++ )
	{
		std::vector<double> errors = EvalMatDeriv( transformedEsts[i] );
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
		std::vector<double> errors = EvaluateCostDerivative<GLL>( likelihoods[i] );
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
	for( unsigned int i = 0; i < popSize; i++ )
	{
		PenalizedGLL penalizedCost( likelihoods[i], 1E-3 );
		std::vector<double> errors = EvaluateCostDerivative<PenalizedGLL>( penalizedCost );
		iter = std::max_element( errors.begin(), errors.end() );
		acc += *iter;
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;
	std::cout << "Avg max error: " << acc / popSize << std::endl;

}	


