#pragma once

#include "percepto/PerceptoTypes.hpp"
#include <cmath>

namespace percepto
{

// Simply passes input through
class NullActivation
{
public:

	typedef double InputType;
	typedef double OutputType;

	NullActivation();

	OutputType Evaluate( const InputType& input ) const;

	OutputType Derivative( const InputType& input ) const;

};

}