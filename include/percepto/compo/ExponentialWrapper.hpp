#pragma once

#include "percepto/compo/BackpropInfo.hpp"

namespace percepto
{

/*! \brief Exponential regressor that takes a vector output and passes it
 * element-wise through an exponential. */
template <typename Regressor>
class ExponentialWrapper
{
public:

	typedef Regressor BaseType;
	typedef typename BaseType::InputType InputType;
	typedef typename BaseType::OutputType OutputType;

	/*! \brief Creates an exponential regressor around a base regressor. Stores
	 * a reference to the regressor. */
	ExponentialWrapper( BaseType& r )
	: _regressor( r ) {}

	unsigned int InputDim() const { return _regressor.InputDim(); }
	unsigned int OutputDim() const { return _regressor.OutputDim(); }
	unsigned int ParamDim() const { return _regressor.ParamDim(); }

	void SetParamsVec( const VectorType& vec ) 
	{
		_regressor.SetParamsVec( vec );
	}

	VectorType GetParamsVec() const
	{
		return _regressor.GetParamsVec();
	}

	// TODO Remove extraneous forward passes
	BackpropInfo Backprop( const InputType& input, const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		BackpropInfo midInfo;
		VectorType mid = _regressor.Evaluate( input );

		MatrixType dydx = MatrixType::Zero( OutputDim(), OutputDim() );
		for( unsigned int i = 0; i < OutputDim(); i++ )
		{
			dydx(i,i) = std::exp( mid(i) );
		}
		midInfo.dodx = nextInfo.dodx * dydx;

		return _regressor.Backprop( input, midInfo );
	}

	OutputType Evaluate( const InputType& input ) const
	{
		return _regressor.Evaluate( input ).array().exp().matrix();
	}

private:

	BaseType& _regressor;

};

}