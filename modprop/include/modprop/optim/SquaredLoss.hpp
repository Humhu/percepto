#pragma once

#include "modprop/compo/Interfaces.h"
#include <iostream>

namespace percepto
{

/*! \brief Provides a squared loss on a vector regression target. */
// TODO Make difference operator a template parameter
template <typename Data>
class SquaredLoss
: public Source<double>
{
public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	typedef Source<Data> InputSourceType;
	typedef Sink<Data> SinkType;
	typedef Source<double> OutputSourceType;
	typedef Data TargetType;
	typedef double OutputType;

	SquaredLoss() 
	: _input( this ), _scale( 1.0 ) {}

	SquaredLoss( const SquaredLoss& other ) 
	: _input( this ), _scale( other._scale ) {}

	void SetSource( InputSourceType* r ) { r->RegisterConsumer( &_input ); }
	void SetTarget( const TargetType& target ) { _target = target; }
	void SetScale( double s ) { _scale = s; }

	virtual unsigned int OutputDim() const { return 1; }

	virtual void Foreprop()
	{
		VectorType err = _input.GetInput() - _target;
		OutputSourceType::SetOutput( 0.5 * err.dot( err ) * _scale );
		OutputSourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		VectorType err = _input.GetInput() - _target;
		if( nextDodx.size() == 0 )
		{
			_input.Backprop( _scale * err.transpose() );
		}
		else
		{
			_input.Backprop( _scale * nextDodx * err.transpose() );
		}
	}

private:

	SinkType _input;
	TargetType _target;
	double _scale;
};

template <>
class SquaredLoss<double>
: public Source<double>
{
public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	typedef Source<double> InputSourceType;
	typedef Sink<double> SinkType;
	typedef Source<double> OutputSourceType;
	typedef double TargetType;
	typedef double OutputType;

	SquaredLoss() 
	: _input( this ), _scale( 1.0 ) {}

	SquaredLoss( const SquaredLoss& other ) 
	: _input( this ), _scale( other._scale ) {}

	void SetSource( InputSourceType* r ) { r->RegisterConsumer( &_input ); }
	void SetTarget( const TargetType& target ) { _target = target; }
	void SetScale( double s ) { _scale = s; }

	virtual unsigned int OutputDim() const { return 1; }

	virtual void Foreprop()
	{
		double err = _input.GetInput() - _target;
		OutputSourceType::SetOutput( 0.5 * err * err * _scale );
		OutputSourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		
		double err = _input.GetInput() - _target;
		MatrixType thisDodx( 1, 1 );
		thisDodx(0,0) = _scale * err;
		if( nextDodx.size() == 0 )
		{
			_input.Backprop( thisDodx );
		}
		else
		{
			_input.Backprop( nextDodx * thisDodx);
		}
	}

private:

	SinkType _input;
	double _target;
	double _scale;
};

}