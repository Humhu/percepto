#pragma once

#include "percepto/PerceptoTypes.h"
#include "percepto/BackpropInfo.hpp"

namespace percepto
{

/*! \brief Exponential regressor that takes a vector output and passes it
 * element-wise through an exponential. */
template <typename Regressor>
class ExponentialWrapper
{
public:

	typedef Regressor BaseType;
	typedef typename BaseType::ParamType ParamType;
	typedef typename BaseType::InputType InputType;
	typedef typename BaseType::OutputType OutputType;

	static ParamType create_zeros( unsigned int inputDim,
	                               unsigned int outputDim )
	{
		return BaseType::create_zeros( inputDim, outputDim );
	}

	/*! \brief Creates an exponential regressor around a base regressor. Makes
	 * a copy of the regressor. */
	ExponentialWrapper( const BaseType& r )
	: _regressor( r ) {}

	ExponentialWrapper( const ParamType& p )
	: _regressor( p ) {}

	unsigned int InputDim() const { return _regressor.InputDim(); }
	unsigned int OutputDim() const { return _regressor.OutputDim(); }
	unsigned int ParamDim() const { return _regressor.ParamDim(); }

	BaseType& GetRegressor() { return _regressor; }

	void SetParams( const ParamType& p ) 
	{ 
		_regressor.SetParams( p ); 
	}

	void SetParamsVec( const VectorType& vec ) 
	{
		_regressor.SetParamsVec( vec );
	}

	ParamType GetParams() const 
	{ 
		return _regressor.GetParams(); 
	}

	VectorType GetParamsVec() const
	{
		return _regressor.GetParamsVec();
	}

	// TODO Remove extraneous forward passes
	void Backprop( const InputType& input, const BackpropInfo& nextInfo,
	               BackpropInfo& thisInfo )
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

		_regressor.Backprop( input, midInfo, thisInfo );
	}

	OutputType Evaluate( const InputType& input ) const
	{
		return _regressor.Evaluate( input ).array().exp().matrix();
	}

private:

	BaseType _regressor;

};

}