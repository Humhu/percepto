#pragma once

#include "percepto/PerceptoTypes.hpp"

namespace percepto
{

// TODO Set zero x and y, active/inactive slopes
class HingeActivation
{
public:

	typedef double InputType;
	typedef double OutputType;

	HingeActivation( double activeSlope = 1.0, double inactiveSlope = 0.0 )
	: _activeSlope( activeSlope ), _inactiveSlope( inactiveSlope ) {}

	OutputType Evaluate( const InputType& input ) const
	{
		if( input > 0 ) { return input * _activeSlope; }
		else { return input * _inactiveSlope; }
	}

	OutputType Derivative( const InputType& input ) const
	{
		if( input > 0 ) { return _activeSlope; }
		else { return _inactiveSlope; }
	}

private:

	double _activeSlope;
	double _inactiveSlope;

};

}