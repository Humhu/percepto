#pragma once

#include "percepto/compo/Interfaces.h"

#include <iostream>
namespace percepto
{

template <typename DataType>
class ScaleWrapper
: public Source<DataType>
{
public:

	typedef Source<DataType> SourceType;
	typedef Sink<DataType> SinkType;

	ScaleWrapper()
	: _input( this ) {}

	ScaleWrapper( const ScaleWrapper& other )
	: _input( this ), _scale( other.scale ) {}

	void SetSource( SourceType* s )
	{
		s->RegisterConsumer( &_input );
	}

	void SetScale( double w )
	{
		_scale = w;
	}

	virtual void Foreprop()
	{
		SourceType::SetOutput( _input.GetInput() * _scale );
		SourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		if( nextDodx.size() == 0 )
		{
			// TODO somehow...
		}
		else
		{
			_input.Backprop( nextDodx * _scale );
		}
	}

private:

	double _scale;
	SinkType _input;

};

}