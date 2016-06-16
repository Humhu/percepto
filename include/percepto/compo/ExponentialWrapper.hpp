#pragma once

#include "percepto/compo/Interfaces.h"

namespace percepto
{

// TODO Matrix version?
/*! \brief Exponential regressor that takes a vector output and passes it
 * element-wise through an exponential. */
template <typename DataType>
class ExponentialWrapper
: public Source<VectorType>
{
public:

	typedef DataType InputType;
	typedef DataType OutputType;
	typedef Source<DataType> SourceType;
	typedef Sink<DataType> SinkType;

	ExponentialWrapper() 
	: _input( this ) {}

	ExponentialWrapper( const ExponentialWrapper& other ) 
	: _input( this ) {}

	// TODO Make a reference?
	void SetSource( SourceType* in ) { in->RegisterConsumer( &_input ); }

	//virtual unsigned int OutputDim() const { return _base->OutputDim(); }

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		// std::cout << "ExponentialWrapper backprop" << std::endl;
		
		DataType mid = _input.GetInput();

		MatrixType dydx = MatrixType::Zero( mid.size(), mid.size() );
		for( unsigned int i = 0; i < mid.size(); i++ )
		{
			dydx(i,i) = std::exp( mid(i) );
		}
		MatrixType thisDodx = nextDodx * dydx;
		_input.Backprop( thisDodx );
	}

	virtual void Foreprop()
	{
		SourceType::SetOutput( _input.GetInput().array().exp().matrix() );
		SourceType::Foreprop();
	}

private:

	SinkType _input;

};

}