#pragma once

#include "percepto/compo/Parametric.hpp"
#include <iostream>

namespace percepto
{

/*! \brief A cost function that adds a weighted L2-norm to an
 * existing cost. For maximization tasks, the weight should be negative. */
class ParameterL2Cost
{
public:

	typedef ScalarType OutputType;

	ParameterL2Cost( Parametric& b, ScalarType w )
	: _base( b ), _w( w )
	{}

	unsigned int OutputDim() const { return 1; }

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		assert( nextDodx.cols() == OutputDim() );

		// Add the L2 cost into the dodw
		VectorType current = _base.GetParamsVec();
		MatrixType thisDodw = _w * nextDodx * current.transpose();
		// MatrixType thisDodw( nextDodx.rows(), current.size() );
		// for( unsigned int i = 0; i < nextDodx.rows(); i++ )
		// {
		// 	thisDodw.row(i) = nextDodx(i,0) * ( _w * current.array() ).matrix().transpose();
		// }
		_base.AccumulateWeightDerivs( thisDodw );
		return thisDodw;
	}

	OutputType Evaluate() const
	{
		VectorType current = _base.GetParamsVec();
		return 0.5 * _w * current.dot( current );
	}

private:

	Parametric& _base;
	ScalarType _w;

};


}