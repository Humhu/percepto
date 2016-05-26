#pragma once

#include "percepto/BackpropInfo.hpp"
#include <memory>

namespace percepto
{

/*! \brief A regressor that multiplies the output of a base
 * regressor on the left and right as:
 * out = transform * base_out * transform.transpose() + offset;
 *  Wraps the base regressor via reference, not copying. */
template <typename Regressor>
class AffineWrapper
{

public:

	typedef Regressor RegressorType;
	typedef typename RegressorType::ParamType ParamType;
	typedef MatrixType OutputType;
	struct InputType
	{
		MatrixType transform;
		MatrixType offset;
		typename RegressorType::InputType baseInput;
	};

	AffineWrapper( RegressorType& r )
	: _regressor( r ) {}

	VectorType CreateWeightVector( double lWeight, double dWeight ) const
	{
		return _regressor.CreateWeightVector( lWeight, dWeight );
	}

	unsigned int InputDim() const { return _regressor.InputDim(); }

	// Note this returns the nominal output dim
	// The true output dim depends on the transforms dimensions
	unsigned int OutputDim() const { return _regressor.OutputDim(); }
	std::pair<unsigned int, unsigned int> OutputSize() const { return _regressor.OutputSize(); }
	unsigned int ParamDim() const { return _regressor.ParamDim(); }

	RegressorType& GetRegressor() { return _regressor; }
	const RegressorType& GetRegressor() const { return _regressor; }

	void SetParams( const ParamType& p ) { _regressor.SetParams( p ); }
	void SetParamsVec( const VectorType& v ) { _regressor.SetParamsVec( v ); }
	ParamType GetParams() const { return _regressor.GetParams(); }
	VectorType GetParamsVec() const { return _regressor.GetParamsVec(); }

	void Backprop( const InputType& input, const BackpropInfo& nextInfo, 
	               BackpropInfo& thisInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		unsigned int oDim = input.transform.rows(); // The actual output dim
		MatrixType dSdx( oDim*oDim, OutputDim() );
		MatrixType d = MatrixType::Zero( OutputSize().first, OutputSize().second );
		for( unsigned int i = 0; i < OutputDim(); i++ )
		{
			d(i) = 1;
			MatrixType temp = input.transform * d * input.transform.transpose();
			dSdx.col(i) = Eigen::Map<VectorType>( temp.data(), temp.size(), 1 );
			d(i) = 0;
		}

		BackpropInfo midInfo;
		midInfo.dodx = nextInfo.dodx * dSdx;
		_regressor.Backprop( input.baseInput, midInfo, thisInfo );
	}

	std::vector<OutputType> AllDerivatives( const InputType& input ) const
	{
		std::vector<OutputType> derivs = _regressor.AllDerivatives( input.baseInput );
		for( unsigned int i = 0; i < derivs.size(); i++ )
		{
			derivs[i] = input.transform * derivs[i] * input.transform.transpose();
		}
		return derivs;
	}

	OutputType Derivative( const InputType& input, unsigned int ind ) const
	{
		return input.transform * _regressor.Derivative( input.baseInput, ind ) *
		       input.transform.transpose();
	}

	OutputType Evaluate( const InputType& input ) const
	{
		return input.transform * _regressor.Evaluate( input.baseInput ) *
		       input.transform.transpose() + input.offset;
	}

private:

	RegressorType& _regressor;

};

}