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
	struct ParamType
	{
		typename RegressorType::ParamType paramA;
		typename RegressorType::ParamType paramB;
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

	void SetParams( const ParamType& p )
	{
		_regA->SetParams( p.paramsA );
		_regB->SetParams( p.paramsB );
	}

	void SetParamsVec( const VectorType& v )
	{
		assert( v.size() == ParamDim() );
		_regA->SetParamsVec( v.block( 0, 0, _regA->ParamsDim(), 1 ) );
		_regB->SetParamsVec( v.block( _regA->ParamsDim(), 0, _regB->ParamsDim(), 1 ) );
	}

	ParamType GetParams() const
	{
		ParamType params;
		params.paramsA = _regA->GetParams();
		params.paramsB = _regB->GetParams();
		return params;
	}

	VectorType GetParamsVec() const
	{
		VectorType vec( ParamDim() );
		vec.block( 0, 0, _regA->ParamDim(), 1 ) = _regA->GetParamsVec();
		vec.block( _regA->ParamDim(), 0, _regB->ParamDim(), 1 ) = _regB->GetParamsVec();
		return vec;
	}

	std::vector<OutputType> AllDerivatives( const InputType& input ) const
	{
		std::vector<OutputType> derivs, derivsA, derivsB;
		derivs.reserve( ParamDim() );
		
		derivsA = _regA->AllDerivatives( input.inputA );
		derivsB = _regB->AllDerivatives( input.inputB );
		for( unsigned int i = 0; i < derivsA.size(); i++ )
		{
			derivs.push_back( derivsA[i] + derivsB[i] );
		}
		return derivs;
	}

	OutputType Derivative( const InputType& input, unsigned int ind ) const
	{
		assert( ind < ParamDim() );
		if( ind < _regA.ParamDim() )
		{
			return _regA.Derivative( input.inputA, ind );
		}
		else
		{
			return _regB.Derivative( input.inputB, ind - _regA.ParamDim() );
		}
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