#pragma once

#include "percepto/PerceptoTypes.h"
#include "percepto/MatrixUtils.hpp"

namespace percepto
{

// TODO Runtime dimension checking?
/** 
 * \brief A wrapper that adds two bases together. 
 */
template <typename Base>
class AdditiveWrapper
{
public:

	typedef Base BaseType;
	typedef typename BaseType::OutputType OutputType;
	struct InputType
	{
		typename BaseType::InputType inputA;
		typename BaseType::InputType inputB;
	};

	AdditiveWrapper( const BaseType& baeA, const BaseType& baseB )
	: _baseA( baseA ), _baseB( baseB )
	{}

	unsigned int InputDim() const { return _baseA.InputDim() + _baseB.InputDim(); }
	unsigned int OutputDim() const { return _baseA.OutputDim() + _baseB.OutputDim(); }
	unsigned int ParamDim() const { return _baseA.ParamDim() + _baseB.ParamDim(); }

	void SetParamsVec( const VectorType& v )
	{
		assert( v.size() == ParamDim() );
		_baseA->SetParamsVec( v.block( 0, 0, _baseA->ParamsDim(), 1 ) );
		_baseB->SetParamsVec( v.block( _baseA->ParamsDim(), 0, _baseB->ParamsDim(), 1 ) );
	}

	VectorType GetParamsVec() const
	{
		return ConcatenateVer( _baseA->GetParamsVec(), _baseB->GetParamsVec() );
	}

	OutputType Evaluate( const InputType& input ) const
	{
		return _baseA.Evaluate( inputA ) + _baseB.Evaluate( inputB );
	}

private:

	BaseType& _baseA;
	BaseType& _baseB;
};

}