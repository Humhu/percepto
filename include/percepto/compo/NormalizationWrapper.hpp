#pragma once

#include "percepto/compo/Interfaces.h"

#include <iostream>

namespace percepto
{

/*! \brief Takes a vector and normalizes it so it has unit L1 norm. */
class L1NormalizationWrapper
: public Source<VectorType>
{
public:

	typedef VectorType InputType;
	typedef VectorType OutputType;
	typedef Source<VectorType> SourceType;
	typedef Sink<VectorType> SinkType;

	L1NormalizationWrapper()
	: _input( this ) {}

	L1NormalizationWrapper( const L1NormalizationWrapper& other )
	: _input( this ) {}

	void SetSource( SourceType* s )
	{
		s->RegisterConsumer( &_input );
	}

	virtual void Foreprop()
	{
		VectorType in = _input.GetInput();
		_z = in.sum();
		SourceType::SetOutput( in / _z );
		SourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		VectorType v = _input.GetInput();

		MatrixType dody = nextDodx;
		if( nextDodx.size() == 0 )
		{
			dody = MatrixType::Identity( v.size(), v.size() );
		}

		MatrixType dydx( v.size(), v.size() );

		double inv_z = 1.0/_z;
		VectorType v_z2 = v / (_z*_z);
		for( unsigned int i = 0; i < v.size(); ++i )
		{
			dydx.row(i).setConstant( -v_z2(i) );
			dydx(i,i) = inv_z - v_z2(i);
		}

		MatrixType thisDodx = dody * dydx;
		_input.Backprop( thisDodx );
	}

private:

	double _z;
	SinkType _input;

};

}