#include "percepto/SigmoidActivation.h"

namespace percepto
{

SigmoidActivation::SigmoidActivation() {}

SigmoidActivation::OutputType 
SigmoidActivation::Evaluate( const InputType& input ) const
{
	return 1.0 / ( 1.0 + std::exp( -input) );
}

SigmoidActivation::OutputType 
SigmoidActivation::Derivative( const InputType& input ) const
{
	double p = std::exp( input );
	double d = 1 + p;
	return p / ( d*d );
}

}