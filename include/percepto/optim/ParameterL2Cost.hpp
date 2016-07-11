#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/compo/Parametric.hpp"
#include <iostream>

namespace percepto
{

/*! \brief A cost function that adds a weighted L2-norm to an
 * existing cost. For maximization tasks, the weight should be negative. */
class ParameterL2Cost
: public Source<double>
{
public:

	typedef Source<double> SourceType;
	typedef ScalarType OutputType;

	ParameterL2Cost()
	: _params( nullptr ), _w( 1.0 ) {}

	void SetWeight( ScalarType w ) { _w = w; }
	void SetParameters( Parameters* p ) { _params = p; }

	virtual unsigned int OutputDim() const { return 1; }

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		// clock_t start = clock();
		// Add the L2 cost into the dodw
		VectorType current = _params->GetParamsVec();
		MatrixType thisDodw;
		if( nextDodx.size() == 0 )
		{
			thisDodw = _w * current.transpose();
		}
		else
		{
			thisDodw = _w * nextDodx * current.transpose();
		}
		_params->AccumulateDerivs( thisDodw );
		// std::cout << "L2 backprop: " << ((double) clock() - start )/CLOCKS_PER_SEC;
	}

	virtual void Foreprop()
	{
		VectorType current = _params->GetParamsVec();
		SourceType::SetOutput( 0.5 * _w * current.dot( current ) );
		SourceType::Foreprop();
	}

private:

	Parameters* _params;
	ScalarType _w;

};


}