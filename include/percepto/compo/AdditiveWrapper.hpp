#pragma once

#include "percepto/compo/Interfaces.h"
#include <iostream>

namespace percepto
{

/*! \brief Adds two sources together. */
template <typename DataType>
class AdditiveWrapper
: public Source<DataType>
{
public:

	typedef DataType InputType;
	typedef DataType OutputType;
	typedef Source<DataType> SourceType;
	typedef Sink<DataType> SinkType;

	AdditiveWrapper() 
	: _inputA( this ), _inputB( this ) {}

	AdditiveWrapper( const AdditiveWrapper& other )
	: _inputA( this ), _inputB( this ) {}

	void SetSourceA( SourceType* a ) 
	{ 
		a->RegisterConsumer( &_inputA );
	}
	void SetSourceB( SourceType* b ) 
	{ 
		b->RegisterConsumer( &_inputB );
	}

	virtual void Foreprop()
	{
		if( _inputA.IsValid() && _inputB.IsValid() )
		{
			SourceType::SetOutput( _inputA.GetInput() + _inputB.GetInput() );
			SourceType::Foreprop();
		}
	}

	virtual void Backprop( const MatrixType& nextDodx )
	{
		// std::cout << "Additive wrapper backprop" << std::endl;
		_inputA.Backprop( nextDodx );
		_inputB.Backprop( nextDodx );
	}

private:

	SinkType _inputA;
	SinkType _inputB;
};

}