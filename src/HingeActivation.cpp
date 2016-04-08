#include "percepto/HingeActivation.h"

namespace percepto
{

HingeActivation::HingeActivation( double activeSlope, double inactiveSlope ) 
: _activeSlope( activeSlope ), _inactiveSlope( inactiveSlope ) {}

HingeActivation::OutputType HingeActivation::Evaluate( const InputType& input ) const
{
	if( input > 0 ) { return input * _activeSlope; }
	else { return input * _inactiveSlope; }
}

HingeActivation::OutputType HingeActivation::Derivative( const InputType& input ) const
{
	if( input > 0 ) { return _activeSlope; }
	else { return _inactiveSlope; }
}

}