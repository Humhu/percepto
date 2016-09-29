#pragma once

#include "modprop/compo/Interfaces.h"
#include <iostream>

namespace percepto
{

// Converts a vector into a diagonal matrix
class DiagonalWrapper
: public Source<MatrixType>
{
public:

	typedef VectorType InputType;
	typedef MatrixType OutputType;
	typedef Source<VectorType> InputSourceType;
	typedef Source<MatrixType> OutputSourceType;
	typedef Sink<VectorType> SinkType;

	DiagonalWrapper()
	: _input( this ) {}

	DiagonalWrapper( const DiagonalWrapper& other )
	: _input( this ) {}

	void SetSource( InputSourceType* s )
	{
		s->RegisterConsumer( &_input );
	}

	virtual void Foreprop()
	{
		VectorType input = _input.GetInput();
		DiagonalType D( input.size() );
		D.diagonal() = input;
		OutputSourceType::SetOutput( D );
		OutputSourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		unsigned int inputDim = _input.GetInput().size();
		unsigned int sysOutDim = nextDodx.rows();

		MatrixType thisDodx( sysOutDim, inputDim );
		for( unsigned int i = 0; i < inputDim; ++i )
		{
			thisDodx.col(i) = nextDodx.col( i * (inputDim+1) );
		}
		_input.Backprop( thisDodx );
	}

private:

	typedef Eigen::DiagonalMatrix<double,Eigen::Dynamic> DiagonalType;

	SinkType _input;
};

}