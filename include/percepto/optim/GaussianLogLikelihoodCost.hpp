#pragma once

#include "percepto/compo/Interfaces.h"
#include <boost/foreach.hpp>
#include <Eigen/Cholesky>
#include <cmath>
#include <iostream>

namespace percepto
{

/*! \brief Calculate the log-likelihood of a sample under a zero-mean Gaussian 
 * distribution with specified covariance. */
inline ScalarType GaussianLogLikelihood( const VectorType& x, 
                                         const MatrixType& cov )
{
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
class GaussianLogLikelihoodCost 
: public Source<double>
{
public:
	
	typedef Source<double> OutputSourceType;
	typedef Source<MatrixType> InputSourceType;
	typedef Sink<MatrixType> SinkType;
	typedef VectorType SampleType;
	typedef ScalarType OutputType;

	/*! \brief Create a cost representing the log likelihood under the matrix
	 * outputted by the regressor. */
	GaussianLogLikelihoodCost() 
	: _input( this ) {}

	GaussianLogLikelihoodCost( const GaussianLogLikelihoodCost& other )
	: _input( this ), _sample( other._sample ) {}

	void SetSource( InputSourceType* s ) { s->RegisterConsumer( &_input ); }
	void SetSample( const SampleType& sample ) { _sample = sample; }
	const SampleType& GetSample() const { return _sample; }

	/*! \brief Computes the log-likelihood of the input sample using the
	 * covariance generated from the input features. */
	virtual void Foreprop()
	{
		double out = -GaussianLogLikelihood( _sample, _input.GetInput() );
		OutputSourceType::SetOutput( out );
		OutputSourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		MatrixType cov = _input.GetInput();
		Eigen::LDLT<MatrixType> ldlt( cov );
		VectorType errSol = ldlt.solve( _sample );
		MatrixType I = MatrixType::Identity( cov.rows(), cov.cols() );
		MatrixType dydS = 0.5 * ldlt.solve( I - _sample * errSol.transpose() );
		Eigen::Map<MatrixType> dydSVec( dydS.data(), 1, dydS.size() );

		if( nextDodx.size() == 0 )
		{
			// std::cout << "GLL: dydSVec: " << dydSVec << std::endl;
			_input.Backprop( dydSVec );
		}
		else
		{
			// std::cout << "GLL: nextDodx * dydSVec: " << nextDodx * dydSVec << std::endl;
			_input.Backprop( nextDodx * dydSVec );
		}
	}

private:
	
	SinkType _input;
	SampleType _sample;
	
};

}

