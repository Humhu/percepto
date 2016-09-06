#pragma once

#include "modprop/compo/Interfaces.h"
#include <iostream>

namespace percepto
{

template <typename Data>
class ElementWrapper
: public Source<double>
{
public:

	typedef Data InputType;
	typedef double OutputType;
	typedef Source<InputType> InputSourceType;
	typedef Source<OutputType> OutputSourceType;
	typedef Sink<InputType> SinkType;

	ElementWrapper()
	: _input( this ) {}

	ElementWrapper( const ElementWrapper& other )
	: _ind( other._ind ), _input( this ) {}

	void SetIndex( unsigned int ind )
	{
		_ind = ind;
	}

	void SetSource( InputSourceType* s )
	{
		s->RegisterConsumer( &_input );
	}

	virtual void Foreprop()
	{
		OutputSourceType::SetOutput( _input.GetInput()( _ind ) );
		OutputSourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != 1 )
		{
			throw std::runtime_error( "ElementWrapper: Backprop dimension error." );
		}

		unsigned int inputDim = _input.GetInput().size();
		unsigned int sysOutDim = nextDodx.rows();
		MatrixType thisDodx = MatrixType::Zero( sysOutDim, inputDim );
		thisDodx.block( 0, _ind, sysOutDim, 1 ) = nextDodx;
		_input.Backprop( thisDodx );
	}

private:

	unsigned int _ind;
	SinkType _input;
};

}