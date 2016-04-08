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

	HingeActivation( double activeSlope = 1.0, double inactiveSlope = 0.0 );

	OutputType Evaluate( const InputType& input ) const;

	OutputType Derivative( const InputType& input ) const;

private:

	double _activeSlope;
	double _inactiveSlope;

};

}