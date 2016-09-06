#pragma once

#include "modprop/compo/Interfaces.h"
#include <boost/foreach.hpp>
#include <Eigen/Cholesky>
#include <cmath>
#include <iostream>

namespace percepto
{

/*! \brief Calculate the log-likelihood of a sample under a zero-mean Gaussian 
 * distribution with specified inverse covariance. */
inline ScalarType GaussianLogPDF( const VectorType& x, 
                                  const MatrixType& info )
{
	// TODO Does not check squareness or PD-ness of cov!
	Eigen::LDLT<MatrixType> ldlt( info );
	
	unsigned int n = info.rows();
	double logdet = 0;
	for( unsigned int i = 0; i < ldlt.vectorD().size(); ++i )
	{
		logdet += std::log( ldlt.vectorD()(i) );;
	}
	return 0.5*( logdet - x.transpose() * info * x - n*std::log( 2*M_PI ) );
}

/*! \brief Represents a cost function calculated as the log likelihood of a
 * set of samples drawn from a zero mean Gaussian with the inverse covariance 
 * specified per sample by a MatrixBase. Returns negative log-likelihood since
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
	: _input( this ), _initialized( false ) {}

	GaussianLogLikelihoodCost( const GaussianLogLikelihoodCost& other )
	: _input( this ), _sample( other._sample ), _initialized( false ) {}

	void SetSource( InputSourceType* s ) { s->RegisterConsumer( &_input ); }
	void SetSample( const SampleType& sample ) { _sample = sample; }
	const SampleType& GetSample() const { return _sample; }

	/*! \brief Computes the log-likelihood of the input sample using the
	 * covariance generated from the input features. */
	virtual void Foreprop()
	{
		double out = -GaussianLogPDF( _sample, _input.GetInput() );
		_initialized = false;
		OutputSourceType::SetOutput( out );
		OutputSourceType::Foreprop();
	}

// TODO Fix to use info
	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		MatrixType cov = _input.GetInput();
		if( !_initialized )
		{
			_ldlt = Eigen::LDLT<MatrixType>( cov );
			_initialized = true;
		}
		VectorType errSol = _ldlt.solve( _sample );
		MatrixType I = MatrixType::Identity( cov.rows(), cov.cols() );
		MatrixType dydS = 0.5 * ( _ldlt.solve( I ) -  errSol * errSol.transpose() );
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
	Eigen::LDLT<MatrixType> _ldlt;
	bool _initialized;

	
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
	: _info( this ), _sample( this ), _initialized( false ) {}

	DynamicGaussianLogLikelihoodCost( const DynamicGaussianLogLikelihoodCost& other )
	: _info( this ), _sample( this ), _initialized( false ) {}

	void SetInfoSource( MatrixSourceType* s ) { s->RegisterConsumer( &_info ); }
	void SetSampleSource( VectorSourceType* s ) { s->RegisterConsumer( &_sample ); }

	/*! \brief Computes the log-likelihood of the input sample using the
	 * covariance generated from the input features. */
	virtual void Foreprop()
	{
		if( _info.IsValid() && _sample.IsValid() )
		{
			double out = -GaussianLogPDF( _sample.GetInput(), _info.GetInput() );
			_initialized = false;
			OutputSourceType::SetOutput( out );
			OutputSourceType::Foreprop();
		}
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		// clock_t start = clock();

		MatrixType info = _info.GetInput();
		VectorType sample = _sample.GetInput();

		if( !_initialized )
		{
			Eigen::LDLT<MatrixType> ldlt = Eigen::LDLT<MatrixType>( info );
			// VectorType errSol = ldlt.solve( sample );
			// VectorType errSol = info * sample;

			// w.r.t. inno
			_dydx = (info * sample).transpose();
			// _dydx = VectorType::Zero( errSol.size() ).transpose();

			// w.r.t. cov
			MatrixType I = MatrixType::Identity( info.rows(), info.cols() );
			_dydS = -0.5 * ldlt.solve( I ) + 0.5 * sample * sample.transpose();

			_initialized = true;
		}
		
		Eigen::Map<MatrixType> dydSVec( _dydS.data(), 1, _dydS.size() );

		// std::cout << "GLL backprop: " << ((double) clock() - start)/CLOCKS_PER_SEC << std::endl;;

		if( nextDodx.size() == 0 )
		{
			_info.Backprop( dydSVec );
			_sample.Backprop( _dydx );
		}
		else
		{
			_info.Backprop( nextDodx * dydSVec );
			MatrixType temp( 1, _dydx.size() ); // Not sure why, matrix vector conversion behaves weirdly
			temp.row(0) = _dydx;
			_sample.Backprop( nextDodx * temp ); // TODO There is a weird broadcasting bug here
		}

		// std::cout << "GLL return: " << ((double) clock() - start)/CLOCKS_PER_SEC << std::endl;;

	}

private:
	
	Sink<MatrixType> _info;
	Sink<VectorType> _sample;

	bool _initialized;
	MatrixType _dydS;
	VectorType _dydx;
	
};

}

