#pragma once

#include "modprop/compo/Interfaces.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/random_device.hpp>

namespace percepto
{

class DropoutLayer :
public Source<VectorType>
{
public:

	enum DropoutMode
	{
		DROPOUT_ENABLE,
		DROPOUT_DISABLE
	};

	typedef Source<VectorType> SourceType;
	typedef VectorType InputType;
	typedef VectorType OutputType;

	DropoutLayer() 
	: _inputPort( this ),
	  _mode( DROPOUT_DISABLE )
	{}

	DropoutLayer( double p ) 
	: _inputPort( this ), _distribution( p ),
	  _mode( DROPOUT_DISABLE ), _p( p )
	{}

	DropoutLayer( const DropoutLayer& other )
	: _inputPort( this ), _distribution( other._distribution ),
	  _p( other._P )
	{
		SetDropoutMode( other._mode );
	}

	void SetDropoutMode( void DropoutMode mode )
	{
		_mode = mode;
		if( _mode == DROPOUT_ENABLE )
		{
			boost::random_device rng;
			_distribution.seed( rng );
		}
	}

	void Resample()
	{
		if( _mode == DROPOUT_DISABLE )
		{
			_mask.SetConstant( _p );
		}
		else
		{
			for( unsigned int i = 0; i < _mask.size(); ++i )
			{
				_mask(i) = _distribution( _generator ) ? 1.0 : 0.0;
			}
		}
	}

	virtual void Foreprop()
	{
		VectorType input = _inputPort.GetInput();
		
		if( _mask.size() == 0 )
		{
			_mask = VectorType::Ones( input.size() );
		}

		VectorType output = ( input.array() * _mask.array() ).matrix();
		SourceType::SetOutput( output );
		SourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		MatrixType dodx = nextDodx;
		for( unsigned int i = 0; i < dodx.rows(); ++i )
		{
			VectorType row = dodx.row(i);
			dodx.row(i) = ( row.array() * _mask.array() ).
		}
	}

private:

	Sink<VectorType> _inputPort;
	VectorType _mask;
	DropoutMode _mode;
	double _p;

	boost::mt19937 _generator;
	boost::bernoulli_distribution<double> _distribution;

};

}