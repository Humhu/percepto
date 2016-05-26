#pragma once

#include "percepto/PerceptoTypes.h"
#include <cmath>

namespace percepto
{

// Simply passes input through
class NullActivation
{
public:

	typedef double InputType;
	typedef double OutputType;

	NullActivation() {}

	OutputType Evaluate( const InputType& input ) const
	{
		return input;
	}

	OutputType Derivative( const InputType& input ) const
	{
		return 1.0;
	}

};

}