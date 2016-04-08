#pragma once

#include "percepto/PerceptoTypes.hpp"
#include <cmath>

namespace percepto
{

class SigmoidActivation
{
public:

	typedef double InputType;
	typedef double OutputType;

	SigmoidActivation();

	OutputType Evaluate( const InputType& input ) const;

	OutputType Derivative( const InputType& input ) const;

};

}