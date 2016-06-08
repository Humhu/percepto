#pragma once

#include "percepto/PerceptoTypes.h"
#include <boost/foreach.hpp>
#include <Eigen/Cholesky>
#include <cmath>

namespace percepto
{

/*! \brief Calculate the log-likelihood of a sample under a zero-mean Gaussian 
 * distribution with specified covariance. */
ScalarType GaussianLogLikelihood( const VectorType& x, 
                                  const MatrixType& cov )
{
	// Computing the modified Cholesky decomposition reduces inverse and determinant
	// calculation complexity to n^3 instead of n! or worse
	// TODO Does not check squareness or PD-ness of cov!
	Eigen::LDLT<MatrixType> ldlt( cov );
	
	ScalarType det = ldlt.vectorD().prod();
	ScalarType invProd = x.dot( ldlt.solve( x ) );
	unsigned int n = cov.rows();
	return -0.5*( std::log( det ) + invProd + n*std::log( 2*M_PI ) );
}

/*! \brief Represents a cost function calculated as the log likelihood of a
 * set of samples drawn from a zero mean Gaussian with the covariance specified
 * per sample by a MatrixBase. Returns negative log-likelihood since
 * it is supposed to be a cost. */
template <typename Base>
class GaussianLogLikelihoodCost 
{
public:
	
	typedef Base BaseType;
	typedef VectorType SampleType;
	typedef ScalarType OutputType;

	/*! \brief Create a cost representing the log likelihood under the matrix
	 * outputted by the regressor. */
	GaussianLogLikelihoodCost( BaseType& r, const SampleType& sample )
	: _base( r ), _sample( sample ) {}

	unsigned int OutputDim() const { return 1; }

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "GLL: Backprop dim error." );
		}

		MatrixType cov = _base.Evaluate();
		Eigen::LDLT<MatrixType> ldlt( cov );
		VectorType errSol = ldlt.solve( _sample );
		MatrixType I = MatrixType::Identity( cov.rows(), cov.cols() );
		MatrixType dydS = 0.5 * ldlt.solve( I - _sample * errSol.transpose() );
		Eigen::Map<MatrixType> dydSVec( dydS.data(), 1, dydS.size() );

		// MatrixType thisDodxM( nextDodx.rows(), cov.rows()*cov.cols() );
		// for( unsigned int i = 0; i < nextDodx.rows(); i++ )
		// {
		// 	thisDodxM.row(i) = dydSVec * nextDodx(i,0);
		// }
		MatrixType thisDodx = nextDodx * dydSVec;

		_base.Backprop( thisDodx );
		return thisDodx;
	}

	/*! \brief Computes the log-likelihood of the input sample using the
	 * covariance generated from the input features. */
	OutputType Evaluate() const
	{
		return -GaussianLogLikelihood( _sample, _base.Evaluate() );
	}

	const SampleType& GetSample() const { return _sample; }

private:
	
	BaseType& _base;
	const SampleType _sample;
	
};

}

