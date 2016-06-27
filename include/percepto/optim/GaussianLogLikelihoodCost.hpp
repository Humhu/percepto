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
			_input.Backprop( dydSVec );
		}
		else
		{
			_input.Backprop( nextDodx * dydSVec );
		}
	}

private:
	
	SinkType _input;
	SampleType _sample;
	
};

class DynamicGaussianLogLikelihoodCost 
: public Source<double>
{
public:
	
	typedef Source<double> OutputSourceType;
	typedef Source<MatrixType> MatrixSourceType;
	typedef Source<VectorType> VectorSourceType;
	typedef Sink<MatrixType> SinkType;
	typedef VectorType SampleType;
	typedef ScalarType OutputType;

	/*! \brief Create a cost representing the log likelihood under the matrix
	 * outputted by the regressor. */
	DynamicGaussianLogLikelihoodCost() 
	: _cov( this ), _sample( this ) {}

	DynamicGaussianLogLikelihoodCost( const DynamicGaussianLogLikelihoodCost& other )
	: _cov( this ), _sample( this ) {}

	void SetCovSource( MatrixSourceType* s ) { s->RegisterConsumer( &_cov ); }
	void SetSampleSource( VectorSourceType* s ) { s->RegisterConsumer( &_sample ); }

	/*! \brief Computes the log-likelihood of the input sample using the
	 * covariance generated from the input features. */
	virtual void Foreprop()
	{
		if( _cov.IsValid() && _sample.IsValid() )
		{
			double out = -GaussianLogLikelihood( _sample.GetInput(), _cov.GetInput() );
			OutputSourceType::SetOutput( out );
			OutputSourceType::Foreprop();
		}
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		MatrixType cov = _cov.GetInput();
		VectorType sample = _sample.GetInput();

		Eigen::LDLT<MatrixType> ldlt( cov );
		VectorType errSol = ldlt.solve( sample );

		// w.r.t. inno
		VectorType dydx = 2*errSol.transpose();

		// w.r.t. cov
		MatrixType I = MatrixType::Identity( cov.rows(), cov.cols() );
		MatrixType dydS = 0.5 * ldlt.solve( I - sample * errSol.transpose() );
		Eigen::Map<MatrixType> dydSVec( dydS.data(), 1, dydS.size() );

		if( nextDodx.size() == 0 )
		{
			_cov.Backprop( dydSVec );
			_sample.Backprop( dydx );
		}
		else
		{
			_cov.Backprop( nextDodx * dydSVec );
			MatrixType temp( 1, dydx.size() ); // Not sure why, matrix vector conversion behaves weirdly
			temp.row(0) = dydx;
			_sample.Backprop( nextDodx * temp ); // TODO There is a weird broadcasting bug here
		}
	}

private:
	
	Sink<MatrixType> _cov;
	Sink<VectorType> _sample;
	
};

}

