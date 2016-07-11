#pragma once
#include <iostream>
#include "percepto/compo/Interfaces.h"

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

	// TODO Handle empty nextDodx
	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		// clock_t start = clock();
		// std::cout << "Additive: nextDodx: " << nextDodx << std::endl;
		_inputA.Backprop( nextDodx );
		// if( !SourceType::modName.empty() )
		// {
		// 	std::cout << SourceType::modName << " backprop A return: " << ((double) clock() - start)/CLOCKS_PER_SEC << std::endl;
		// }
		// start = clock();
		_inputB.Backprop( nextDodx );
		// if( !SourceType::modName.empty() )
		// {
		// 	std::cout << SourceType::modName << " backprop B return: " << ((double) clock() - start)/CLOCKS_PER_SEC << std::endl;
		// }
	}

private:

	SinkType _inputA;
	SinkType _inputB;
};

}