#pragma once

#include "percepto/PerceptoTypes.h"

namespace percepto
{

/*! \brief A wrapper that adds two regressors together. Stores
 * the base regressors via references. */
template <typename Regressor>
class AdditiveWrapper
{
public:

	typedef Regressor RegressorType;
	typedef typename RegressorType::OutputType OutputType;
	struct InputType
	{
		typename RegressorType::InputType inputA;
		typename RegressorType::InputType inputB;
	};

	AdditiveWrapper( const RegressorType& regA, const RegressorType& regB )
	: _regA( regA ), _regB( regB )
	{}

	unsigned int InputDim() const { return _regA.InputDim() + _regB.InputDim(); }
	unsigned int OutputDim() const { return _regA.OutputDim() + _regB.OutputDim(); }
	unsigned int ParamDim() const { return _regA.ParamDim() + _regB.ParamDim(); }

	VectorType CreateWeightVector( double lWeight, double dWeight ) const
	{
		VectorType weights( ParamDim() );
		weights.block( ind, 0, _regA->ParamDim(), 1 ) = 
			_regA->CreateWeightVector( lWeight, dWeight );
		weights.block( _regA->ParamDim(), 0, _regB->ParamDim(), 1 ) = 
			_regB->CreateWeightVector( lWeight, dWeight );
		return weights;
	}

	void SetParamsVec( const VectorType& v )
	{
		assert( v.size() == ParamDim() );
		_regA->SetParamsVec( v.block( 0, 0, _regA->ParamsDim(), 1 ) );
		_regB->SetParamsVec( v.block( _regA->ParamsDim(), 0, _regB->ParamsDim(), 1 ) );
	}

	VectorType GetParamsVec() const
	{
		VectorType vec( ParamDim() );
		vec.block( 0, 0, _regA->ParamDim(), 1 ) = _regA->GetParamsVec();
		vec.block( _regA->ParamDim(), 0, _regB->ParamDim(), 1 ) = _regB->GetParamsVec();
		return vec;
	}

	OutputType Evaluate( const InputType& input ) const
	{
		return _regA.Evaluate( inputA ) + _regB.Evaluate( inputB );
	}

private:

	RegressorType& _regA;
	RegressorType& _regB;
};

}