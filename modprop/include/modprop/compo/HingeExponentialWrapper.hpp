#pragma once

#include "modprop/compo/Interfaces.h"

namespace percepto
{

// TODO Matrix version?
/*! \brief Exponential regressor that takes a vector output and passes it
 * element-wise through an exponential, or linear if it passes a threshold. */
template <typename DataType>
class HingeExponentialWrapper
: public Source<VectorType>
{
public:

	typedef DataType InputType;
	typedef DataType OutputType;
	typedef Source<DataType> SourceType;
	typedef Sink<DataType> SinkType;

	HingeExponentialWrapper( double lowerThresh, double upperThresh ) 
	: _input( this ), _lowerThresh( lowerThresh ), _upperThresh( upperThresh ), 
	_lowerSlope( std::exp( lowerThresh ) ), _upperSlope( std::exp( upperThresh ) ),
	_initialized( false ) {}

	HingeExponentialWrapper( const HingeExponentialWrapper& other ) 
	: _input( this ), _lowerThresh( other._lowerThresh ), _upperThresh( other._upperThresh ), 
	_lowerSlope( other._lowerSlope ), _upperSlope( other._upperSlope ),
	_initialized( false ) {}

	// TODO Make a reference?
	void SetSource( SourceType* in ) { in->RegisterConsumer( &_input ); }

	//virtual unsigned int OutputDim() const { return _base->OutputDim(); }

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		// std::cout << "HingeExponentialWrapper backprop" << std::endl;
		
		if( !_initialized )
		{
			DataType mid = _input.GetInput();

			_dydx = MatrixType::Zero( mid.size(), mid.size() );
			for( unsigned int i = 0; i < mid.size(); i++ )
			{
				if( mid(i) > _upperThresh )
				{
					_dydx(i,i) = _upperSlope;
				}
				else if( mid(i) < _lowerThresh )
				{
					_dydx(i,i) = _lowerSlope;
				}
				else
				{
					_dydx(i,i) = std::exp( mid(i) );
				}
			}
			_initialized = true;
		}
		
		MatrixType thisDodx = nextDodx * _dydx;
		_input.Backprop( thisDodx );
	}

	virtual void Foreprop()
	{
		_initialized = false;
		VectorType in = _input.GetInput();
		VectorType out( in.size() );
		for( unsigned int i = 0; i < out.size(); ++i )
		{
			if( in(i) > _upperThresh )
			{
				out(i) = _upperSlope + _upperSlope * (in(i) - _upperThresh);
			}
			else if( in(i) < _lowerThresh )
			{
				out(i) = _lowerSlope + _lowerSlope * (in(i) - _lowerThresh);
				if( out(i) < 0 ) { out(i) = 0; }
			}
			else
			{
				out(i) = std::exp( in(i) );
			}
		}
		// SourceType::SetOutput( _input.GetInput().array().exp().matrix() );
		SourceType::SetOutput( out );
		SourceType::Foreprop();
	}

private:

	SinkType _input;
	double _lowerThresh;
	double _upperThresh;
	double _lowerSlope;
	double _upperSlope; //NOTE Slope is also offset b/c exp
	bool _initialized;
	MatrixType _dydx;

};

}