#pragma once

#include "percepto/compo/Interfaces.h"
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

	virtual void Backprop( const MatrixType& nextDodx )
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

}