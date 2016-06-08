#pragma once

#include "percepto/PerceptoTypes.h"

namespace percepto
{

/*! \brief Exponential regressor that takes a vector output and passes it
 * element-wise through an exponential. */
template <typename Base>
class ExponentialWrapper
{
public:

	typedef Base BaseType;
	typedef typename BaseType::OutputType OutputType;

	/*! \brief Creates an exponential regressor around a base regressor. Stores
	 * a reference to the regressor. */
	ExponentialWrapper( BaseType& r )
	: _base( r ) {}

	unsigned int OutputDim() const { return _base.OutputDim(); }

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "ExponentialWrapper: Backprop dim error." );
		}

		VectorType mid = _base.Evaluate();

		MatrixType dydx = MatrixType::Zero( OutputDim(), OutputDim() );
		for( unsigned int i = 0; i < OutputDim(); i++ )
		{
			dydx(i,i) = std::exp( mid(i) );
		}
		MatrixType thisDodx = nextDodx * dydx;
		return _base.Backprop( thisDodx );
	}

	OutputType Evaluate() const
	{
		return _base.Evaluate().array().exp().matrix();
	}

private:

	BaseType& _base;

};

}