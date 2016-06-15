#pragma once

#include "percepto/PerceptoTypes.h"

namespace percepto
{

// TODO Set zero x and y, active/inactive slopes
class HingeActivation
{
public:

	typedef double InputType;
	typedef double OutputType;

	const double activeSlope;
	const double inactiveSlope;

	HingeActivation( double actSlope = 1.0, double inactSlope = 0.0 )
	: activeSlope( actSlope ), inactiveSlope( inactSlope ) {}

	HingeActivation( const HingeActivation& other )
	: activeSlope( other.activeSlope ), inactiveSlope( other.inactiveSlope ) {}

	OutputType Evaluate( const InputType& input ) const
	{
		if( input > 0 ) { return input * activeSlope; }
		else { return input * inactiveSlope; }
	}

	OutputType Derivative( const InputType& input ) const
	{
		if( input > 0 ) { return activeSlope; }
		else { return inactiveSlope; }
	}

};

}