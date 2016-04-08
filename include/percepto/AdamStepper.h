#pragma once

#include "percepto/PerceptoTypes.hpp"
#include "percepto/PerceptoUtils.h"

namespace percepto
{

struct AdamParameters
{
	// The step size (1E-3)
	double alpha;
	// The first moment memory term (0.9)
	double beta1;
	// The second moment memory term (0.999)
	double beta2;
	// The second moment norm offset (1E-7)
	double epsilon;

	AdamParameters();
};

// Based on work by Kingma and Ba. See (Kingma, Ba 2015)
class AdamStepper
{
public:

	AdamStepper( const AdamParameters& params = AdamParameters() );

	void Reset();

	VectorType GetStep( const VectorType& gradient );

private:

	AdamParameters _params;

	VectorType _m;
	VectorType _v;
	unsigned int _t;

};

}