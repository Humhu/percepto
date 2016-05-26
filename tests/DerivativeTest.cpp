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

typedef TriangularMapping TriMapd;
typedef ReLUNet BaseRegressor;
typedef ExponentialWrapper<BaseRegressor> ExpRegressor;
typedef ModifiedCholeskyWrapper<BaseRegressor, ExpRegressor> ModCholRegressor;

typedef AffineWrapper<ModCholRegressor> AffineModCholReg;

typedef GaussianLogLikelihoodCost<AffineModCholReg> GLLd;

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
std::vector<double> EvalMatDeriv( Regressor& r, 
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
		
		VectorType gi = grad.row(ind);
		Eigen::Map<MatrixType> dS( gi.data(), r.OutputSize().first, r.OutputSize().second );
		MatrixType predOut = output + dS*stepSize;
		double predErr = ( predOut - testOutput ).norm();
		errors[ind] = predErr/change;
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

	unsigned int lFeatDim = 5;
	unsigned int dFeatDim = 5;
	unsigned int lOutDim = triMap.NumPositions();
	unsigned int dOutDim = matDim;
	
	unsigned int lNumHiddenLayers = 1;
	unsigned int lLayerWidth = 10;
	unsigned int dNumHiddenLayers = 1;
	unsigned int dLayerWidth = 10;

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "L Feature dim: " << lFeatDim << std::endl;
	std::cout << "D Feature dim: " << dFeatDim << std::endl;
	std::cout << "L Output dim: " << lOutDim << std::endl;
	std::cout << "D Output dim: " << dOutDim << std::endl;

	//BaseRegressor lReg( BaseRegressor::ParamType::Random( lOutDim, lFeatDim ) );
	HingeActivation relu( 1.0, 1E-3 );
	BaseRegressor lReg = BaseRegressor::create_zeros( lFeatDim, lOutDim, 
	                                                  lNumHiddenLayers, lLayerWidth, relu );
	VectorType params = lReg.GetParamsVec();
	randomize_vector( params );
	lReg.SetParamsVec( params );

	//ExpRegressor expreg( ExpRegressor::ParamType::Random( dOutDim, dFeatDim ) );
	BaseRegressor dReg = BaseRegressor::create_zeros( dFeatDim, dOutDim, 
	                                                  dNumHiddenLayers, dLayerWidth, relu );
	params = dReg.GetParamsVec();
	randomize_vector( params );
	dReg.SetParamsVec( params );

	ExpRegressor expreg( dReg );

	ModCholRegressor mcReg( lReg, expreg, 
	                        1E-3 * MatrixType::Identity( matDim, matDim ) );

	AffineModCholReg amcReg( mcReg );
	// Test derivatives of various functions here
	unsigned int popSize = 100;

	// 1. Generate test set
	std::vector<GLLd> testCosts;
	testCosts.reserve( popSize );
	for( unsigned int i = 0; i < popSize; i++ )
	{
		typedef ModCholRegressor::LRegressorType::InputType LInputType;
		typedef ModCholRegressor::DRegressorType::InputType DInputType;
		
		ModCholRegressor::InputType mcInput;
		mcInput.lInput = LInputType::Random( lFeatDim );
		mcInput.dInput = DInputType::Random( dFeatDim );

		AffineModCholReg::InputType amcInput;
		amcInput.baseInput = mcInput;
		amcInput.transform = MatrixType::Random( matDim, matDim );
		amcInput.offset = MatrixType::Identity( matDim, matDim );

		VectorType sample = VectorType::Random( matDim );
		AffineModCholReg::InputType input = amcInput;

		testCosts.emplace_back( amcInput, sample, amcReg );
	}

	// 2. Test linear gradients
	std::cout << "Testing base gradients..." << std::endl;
	double maxSeen = -std::numeric_limits<double>::infinity();
	for( unsigned int i = 0; i < popSize; i++ )
	{
		const BaseRegressor::InputType& input = testCosts[i]._input.baseInput.lInput;
		std::vector<double> errors = EvalVecDeriv( lReg, input );
		std::vector<double>::iterator iter;
		iter = std::max_element( errors.begin(), errors.end() );
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;

	// 3. Test exponential gradients
	std::cout << "Testing exponential gradients..." << std::endl;
	maxSeen = -std::numeric_limits<double>::infinity();
	for( unsigned int i = 0; i < popSize; i++ )
	{
		const ExpRegressor::InputType& input = testCosts[i]._input.baseInput.dInput;
		std::vector<double> errors = EvalVecDeriv( expreg, input );
		std::vector<double>::iterator iter;
		iter = std::max_element( errors.begin(), errors.end() );
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;

	// 4. Test cholesky gradients
	std::cout << "Testing modified Cholesky gradients..." << std::endl;
	maxSeen = -std::numeric_limits<double>::infinity();
	for( unsigned int i = 0; i < popSize; i++ )
	{
		const ModCholRegressor::InputType& input = testCosts[i]._input.baseInput;
		std::vector<double> errors = EvalMatDeriv( mcReg, input );
		std::vector<double>::iterator iter;
		iter = std::max_element( errors.begin(), errors.end() );
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;

	// 5. Test transformed and damped cholesky gradients
	std::cout << "Testing affine modified Cholesky gradients..." << std::endl;
	maxSeen = -std::numeric_limits<double>::infinity();
	for( unsigned int i = 0; i < popSize; i++ )
	{
		const AffineModCholReg::InputType& input = testCosts[i]._input;
		std::vector<double> errors = EvalMatDeriv( amcReg, input );
		std::vector<double>::iterator iter;
		iter = std::max_element( errors.begin(), errors.end() );
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;

	// 6. Test log-likelihood gradients
	std::cout << "Testing log-likelihood gradients..." << std::endl;
	maxSeen = -std::numeric_limits<double>::infinity();
	for( unsigned int i = 0; i < popSize; i++ )
	{
		std::vector<double> errors = EvaluateCostDerivative<GLLd>( testCosts[i] );
		std::vector<double>::iterator iter;
		iter = std::max_element( errors.begin(), errors.end() );
		if( *iter > maxSeen ) { maxSeen = *iter; }
	}
	std::cout << "Max error: " << maxSeen << std::endl;

}	


