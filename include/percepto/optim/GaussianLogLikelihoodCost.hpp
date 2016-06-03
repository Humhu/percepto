#pragma once

#include "percepto/compo/BackpropInfo.hpp"
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
	 * outputted by the regressor.  Stores references, not value. */
	GaussianLogLikelihoodCost( BaseType& r, const SampleType& sample )
	: _base( r ), _sample( sample ) {}

	unsigned int OutputDim() const { return 1; }
	unsigned int ParamDim() const { return _base.ParamDim(); }

	void SetParamsVec( const VectorType& v )
	{
		_base.SetParamsVec( v );
	}

	VectorType GetParamsVec() const
	{
		return _base.GetParamsVec();
	}

	BackpropInfo Backprop( const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		MatrixType cov = _base.Evaluate();
		Eigen::LDLT<MatrixType> ldlt( cov );
		VectorType errSol = ldlt.solve( _sample );
		MatrixType I = MatrixType::Identity( cov.rows(), cov.cols() );
		MatrixType dydS = ldlt.solve( I - _sample * errSol.transpose() );
		Eigen::Map<VectorType> dydSVec( dydS.data(), dydS.size(), 1 );

		BackpropInfo midInfo;
		midInfo.dodx = MatrixType( nextInfo.SystemOutputDim(), cov.rows()*cov.cols() );
		for( unsigned int i = 0; i < nextInfo.SystemOutputDim(); i++ )
		{
			midInfo.dodx.row(i) = dydSVec * nextInfo.dodx(i);
		}

		return _base.Backprop( midInfo );
	}

	/*! \brief Computes the log-likelihood of the input sample using the
	 * covariance generated from the input features. */
	OutputType Evaluate() const
	{
		return -GaussianLogLikelihood( _sample, _base.Evaluate() );
	}

private:
	
	BaseType& _base;
	const SampleType _sample;
	
};

}

