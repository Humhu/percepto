#pragma once

#include "modprop/compo/Interfaces.h"

namespace percepto
{

template <typename Data>
class DummyLoss
: public Source<double>
{
public:

	typedef Source<Data> InputSourceType;
	typedef Sink<Data> SinkType;
	typedef Source<double> OutputSourceType;

	DummyLoss()
	: _input( this ) {}

	DummyLoss( const DummyLoss& other )
	: _input( this ) {}

	void SetSource( InputSourceType* r ) { r->RegisterConsumer( &_input ); }

	virtual unsigned int OutputDim() const { return 1; }

	virtual void Foreprop()
	{
		OutputSourceType::SetOutput( 0 );
		OutputSourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		Data input = _input.GetInput();
		_input.Backprop( MatrixType::Zero( nextDodx.rows(), input.size() ) );
	}

private:

	SinkType _input;

};

}