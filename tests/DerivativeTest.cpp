#include "percepto/LinearRegressor.hpp"
#include "percepto/ExponentialWrapper.hpp"
#include "percepto/AffineWrapper.hpp"

#include "percepto/pdreg/LowerTriangular.hpp"
#include "percepto/pdreg/ModifiedCholeskyWrapper.hpp"

#include "percepto/GaussianLogLikelihoodCost.hpp"

#include <ctime>
#include <iostream>

using namespace percepto;

typedef TriangularMapping TriMapd;
typedef LinearRegressor LinReg;
typedef ExponentialWrapper<LinReg> ExpReg;
typedef ModifiedCholeskyWrapper<LinReg,ExpReg> ModCholReg;

typedef AffineWrapper<ModCholReg> AffineModCholReg;

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
	typedef typename Regressor::ParamType ParamType;

	OutType output, testOutput;
	VectorType paramsVec = r.GetParamsVec();
	ParamType testParams;
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
	typedef typename Regressor::ParamType ParamType;

	OutType output, testOutput;
	VectorType paramsVec = r.GetParamsVec();
	ParamType testParams;
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
		
		std::cout << "Index " << ind << " output: " << std::endl << testOutput.transpose() << std::endl;

		double change = ( output - testOutput ).norm();
		
		VectorType gi = grad.row(ind);
		Eigen::Map<MatrixType> dS( gi.data(), r.OutputSize().first, r.OutputSize().second );
		MatrixType predOut = output + dS*stepSize;
		std::cout << "pred out: " << std::endl << predOut << std::endl;
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
	std::cout << "nom: " << output << std::endl;

	for( unsigned int ind = 0; ind < cost.ParamDim(); ind++ )
	{
		// Take a step and check the output
		VectorType testParamsVec = paramsVec;
		testParamsVec[ind] += stepSize;
		cost.SetParamsVec( testParamsVec );
		testOutput = cost.Evaluate();
		cost.SetParamsVec( paramsVec );

		std::cout << "Index " << ind << " output: " << testOutput << std::endl;		
		double predOut = output + grad(ind)*stepSize;
		std::cout << "pred out: " << predOut << std::endl;
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
	unsigned int dFeatDim = 1;
	unsigned int linDim = triMap.NumPositions();

	std::cout << "Matrix dim: " << matDim << std::endl;
	std::cout << "L Feature dim: " << lFeatDim << std::endl;
	std::cout << "D Feature dim: " << dFeatDim << std::endl;
	std::cout << "L dim: " << linDim << std::endl;

	LinReg linreg( LinReg::ParamType::Random( linDim, lFeatDim ) );
	ExpReg expreg( LinReg::ParamType::Random( matDim, dFeatDim ) );
	ModCholReg mcReg( linreg, expreg,
	                   1E-3 * MatrixType::Identity( matDim, matDim ) );

	AffineModCholReg amcReg( mcReg );
	// Test derivatives of various functions here
	unsigned int popSize = 1;

	// 1. Generate test set
	std::vector<GLLd> testCosts;
	testCosts.reserve( popSize );
	for( unsigned int i = 0; i < popSize; i++ )
	{
		typedef ModCholReg::LRegressorType::InputType LInputType;
		typedef ModCholReg::DRegressorType::InputType DInputType;
		
		ModCholReg::InputType mcInput;
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
	std::cout << "Testing linear gradients..." << std::endl;
	double maxSeen = -std::numeric_limits<double>::infinity();
	for( unsigned int i = 0; i < popSize; i++ )
	{
		const LinReg::InputType& input = testCosts[i]._input.baseInput.lInput;
		std::vector<double> errors = EvalVecDeriv( linreg, input );
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
		const ExpReg::InputType& input = testCosts[i]._input.baseInput.dInput;
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
		const ModCholReg::InputType& input = testCosts[i]._input.baseInput;
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


