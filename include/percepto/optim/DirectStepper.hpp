#pragma once

#include "percepto/PerceptoTypes.h"

#include <iostream>

namespace percepto
{

struct DirectStepperParameters
{
	// The step size (1E-3)
	double alpha;
	// Whether or not to shrink the step size over time
	bool enableDecay;

	DirectStepperParameters()
	: alpha( 1E-3 ), enableDecay( false ) {}
};


class DirectStepper
{
public:

	/*! \brief Create a new stepper with the specified parameters. */
	DirectStepper( const DirectStepperParameters& params = DirectStepperParameters() )
	: _params( params )
	{
		Reset();
	}

	/*! \brief Reset the stepper state. */
	void Reset()
	{ 
		_t = 0; 
	}

	VectorType GetStep( const VectorType& gradient )
	{
		++_t;
		double step = _params.alpha;
		if( _params.enableDecay )
		{
			step = _params.alpha / std::sqrt( _t );
		}
		return step * gradient;
	}

private:

	DirectStepperParameters _params;
	unsigned int _t;
};

}