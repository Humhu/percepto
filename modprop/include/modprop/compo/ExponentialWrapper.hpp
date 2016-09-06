#pragma once

#include "modprop/compo/Interfaces.h"

namespace percepto
{

// TODO Matrix version?
/*! \brief Exponential regressor that takes a vector output and passes it
 * element-wise through an exponential. */
class ExponentialWrapper
: public Source<VectorType>
{
public:

	typedef VectorType InputType;
	typedef VectorType OutputType;
	typedef Source<VectorType> SourceType;
	typedef Sink<VectorType> SinkType;

	ExponentialWrapper() 
	: _input( this ), _initialized( false ) {}

	ExponentialWrapper( const ExponentialWrapper& other ) 
	: _input( this ), _initialized( false ) {}

	// TODO Make a reference?
	void SetSource( SourceType* in ) { in->RegisterConsumer( &_input ); }

	//virtual unsigned int OutputDim() const { return _base->OutputDim(); }

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		// std::cout << "ExponentialWrapper backprop" << std::endl;
		
		if( !_initialized )
		{
			VectorType mid = _input.GetInput();

			_dydx = MatrixType::Zero( mid.size(), mid.size() );
			for( unsigned int i = 0; i < mid.size(); i++ )
			{
				_dydx(i,i) = std::exp( mid(i) );
			}
			_initialized = true;
		}
		
		MatrixType thisDodx = nextDodx * _dydx;
		_input.Backprop( thisDodx );
	}

	virtual void Foreprop()
	{
		_initialized = false;
		SourceType::SetOutput( _input.GetInput().array().exp().matrix() );
		SourceType::Foreprop();
	}

private:

	SinkType _input;
	bool _initialized;
	MatrixType _dydx;

};

}