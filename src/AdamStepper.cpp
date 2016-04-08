#include "percepto/AdamStepper.h"

namespace percepto
{

AdamParameters::AdamParameters()
: alpha( 1E-3 ), beta1( 0.9 ), beta2( 0.999 ), epsilon( 1E-7 ) {}

AdamStepper::AdamStepper( const AdamParameters& params )
: _params( params )
{
	if( _params.beta1 < 0 || _params.beta1 > 1.0 )
	{
		throw std::runtime_error( "beta1 must be between 0 and 1." );
	}
	if( _params.beta2 < 0 || _params.beta2 > 1.0 )
	{
		throw std::runtime_error( "beta2 must be between 0 and 1." );
	}
}

void AdamStepper::Reset() 
{ 
	_t = 0; 
	_m = VectorType();
}

VectorType AdamStepper::GetStep( const VectorType& gradient )
{
	// Need to initialize zeros
	if( _m.size() == 0 )
	{
		_m = VectorType::Zero( gradient.size() );
		_v = VectorType::Zero( gradient.size() );
	}
	assert( gradient.size() == _m.size() );

	++_t;
	_m = _params.beta1 * _m + (1.0 - _params.beta1 ) * gradient;
	VectorType gradientSq = ( gradient.array() * gradient.array() ).matrix();
	_v = _params.beta2 * _v + (1.0 - _params.beta2 ) * gradientSq;

	VectorType mhat = _m / ( 1.0 - std::pow(_params.beta1, _t) );
	VectorType vhat = _v / ( 1.0 - std::pow(_params.beta2, _t) );

	return ( _params.alpha * mhat.array() / 
	         ( vhat.array().sqrt() + _params.epsilon ) ).matrix();
}


}