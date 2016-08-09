#pragma once

#include "percepto/compo/Interfaces.h"

namespace percepto
{

class SubvectorWrapper
: public Source<VectorType>
{
public:

	typedef VectorType InputType;
	typedef VectorType OutputType;
	typedef Source<VectorType> SourceType;
	typedef Sink<VectorType> SinkType;

	SubvectorWrapper()
	: _input( this ) {}

	SubvectorWrapper( const SubvectorWrapper& other )
	: _startInd( other._startInd ), _length( other._length ),
	_input( this ) {}

	void SetInds( unsigned int startInd, unsigned int len )
	{
		_startInd = startInd;
		_length = len;
	}

	void SetSource( SourceType* s )
	{
		s->RegisterConsumer( &_input );
	}

	virtual void Foreprop()
	{
		SourceType::SetOutput( _input.GetInput().segment( _startInd, _length ) );
		SourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		unsigned int inputDim = _input.GetInput().size();
		unsigned int sysOutDim = nextDodx.rows();
		MatrixType thisDodx = MatrixType::Zero( sysOutDim, inputDim );
		thisDodx.block( 0, _startInd, sysOutDim, _length ) = nextDodx;
		_input.Backprop( thisDodx );
	}

private:

	unsigned int _startInd;
	unsigned int _length;
	SinkType _input;
};

}