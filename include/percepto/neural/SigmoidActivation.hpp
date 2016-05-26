#pragma once

#include "percepto/PerceptoTypes.h"
#include <cmath>

namespace percepto
{

class SigmoidActivation
{
public:

	typedef double InputType;
	typedef double OutputType;

	SigmoidActivation() {}

	OutputType Evaluate( const InputType& input ) const
	{
		return 1.0 / ( 1.0 + std::exp( -input) );
	}

	OutputType Derivative( const InputType& input ) const
	{
		double p = std::exp( input );
		double d = 1 + p;
		return p / ( d*d );
	}

};

}