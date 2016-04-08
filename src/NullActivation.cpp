#include "percepto/NullActivation.h"

namespace percepto
{

NullActivation::NullActivation() {}

NullActivation::OutputType NullActivation::Evaluate( const InputType& input ) const
{
	return input;
}

NullActivation::OutputType NullActivation::Derivative( const InputType& input ) const
{
	return 1.0;
}

}