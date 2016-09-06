#pragma once

#include "modprop/compo/Interfaces.h"
#include "modprop/optim/GaussianLogLikelihoodCost.hpp"

namespace percepto
{

class GaussianLogProbability 
: public Source<double>
{
public:
	
	typedef Source<double> OutputSourceType;
	typedef Source<MatrixType> InfoSourceType;
	typedef Source<VectorType> MeanSourceType;
	typedef Sink<MatrixType> SinkType;
	typedef VectorType SampleType;

	GaussianLogProbability() 
	: _info( this ), _mean( this ), _initialized( false ) {}

	GaussianLogProbability( const GaussianLogProbability& other )
	: _info( this ), _mean( this ), _initialized( false ) {}

	void SetInfoSource( InfoSourceType* s ) { s->RegisterConsumer( &_info ); }
	void SetMeanSource( MeanSourceType* s ) { s->RegisterConsumer( &_mean ); }
	void SetSample( const VectorType& u ) { _sample = u; }

	const VectorType& GetSample() const { return _sample; }

	/*! \brief Computes the log-likelihood of the input sample using the
	 * covariance generated from the input features. */
	virtual void Foreprop()
	{
		if( _info.IsValid() && _mean.IsValid() )
		{
			VectorType x = _sample - _mean.GetInput();
			double out = GaussianLogPDF( x, _info.GetInput() );
			_initialized = false;

			OutputSourceType::SetOutput( out );
			OutputSourceType::Foreprop();
		}
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		MatrixType info = _info.GetInput();
		VectorType x = _sample - _mean.GetInput();

		if( !_initialized )
		{
			Eigen::LDLT<MatrixType> ldlt = Eigen::LDLT<MatrixType>( info );

			// w.r.t. x
			_dydx = -(info * x).transpose();

			// w.r.t. cov
			MatrixType I = MatrixType::Identity( info.rows(), info.cols() );
			_dydS = 0.5 * ldlt.solve( I ) - 0.5 * x * x.transpose();

			_initialized = true;
		}
		
		Eigen::Map<MatrixType> dydSVec( _dydS.data(), 1, _dydS.size() );

		if( !modName.empty() )
		{
			double logprob = GaussianLogPDF( x, _info.GetInput() );
			std::cout << "nextDodx: " << nextDodx << std::endl;
			std::cout << "samp: " << _sample.transpose() << std::endl;
			std::cout << "mean: " << _mean.GetInput().transpose() << std::endl;
			std::cout << "info: " << std::endl << _info.GetInput() << std::endl;
			std::cout << "log prob: " << logprob << std::endl;
			std::cout << "dydMean: " << -_dydx << std::endl;
			std::cout << "dydInfo: " << _dydS << std::endl;
		}

		if( nextDodx.size() == 0 )
		{
			_info.Backprop( dydSVec );
			// NOTE x = sample - mean so we use -dydx here
			_mean.Backprop( -_dydx );
		}
		else
		{
			_info.Backprop( nextDodx * dydSVec );
			MatrixType temp( 1, _dydx.size() ); // Not sure why, matrix vector conversion behaves weirdly
			// NOTE x = sample - mean so we use -dydx here
			temp.row(0) = -_dydx;
			_mean.Backprop( nextDodx * temp ); // TODO There is a weird broadcasting bug here
		}

	}

private:
	
	Sink<MatrixType> _info;
	Sink<VectorType> _mean;
	VectorType _sample;

	bool _initialized;
	MatrixType _dydS;
	VectorType _dydx;
	
};

}