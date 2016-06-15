#pragma once

#include "percepto/compo/Interfaces.h"
#include <iostream>

namespace percepto
{

template <typename DataType>
class DifferenceWrapper
: public Source<DataType>
{
public:

	typedef DataType InputType;
	typedef DataType OutputType;
	typedef Source<DataType> SourceType;
	typedef Sink<DataType> SinkType;

	DifferenceWrapper() 
	: _plus( this ), _minus( this ) {}

	DifferenceWrapper( const DifferenceWrapper& other ) 
	: _plus( this ), _minus( this ) {}

	void SetPlusSource( SourceType* p ) { p->RegisterConsumer( &_plus ); }
	void SetMinusSource( SourceType* m ) { m->RegisterConsumer( &_minus ); }

	// unsigned int OutputDim() const { return _plus->OutputDim(); }

	virtual void Foreprop()
	{
		if( _plus.IsValid() && _minus.IsValid() )
		{
			SourceType::SetOutput( _plus.GetInput() - _minus.GetInput() );
			SourceType::Foreprop();
		}
	}

	virtual void Backprop( const MatrixType& nextDodx )
	{
		// std::cout << "DifferenceWrapper backprop" << std::endl;
		_plus.Backprop( nextDodx );
		_minus.Backprop( -nextDodx );
	}

private:

	SinkType _plus;
	SinkType _minus;
};

}