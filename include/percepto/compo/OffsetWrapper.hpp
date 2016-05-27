#pragma once

#include "percepto/compo/BackpropInfo.hpp"
#include <memory>
#include <stdexcept>

namespace percepto
{

/** 
 * \brief A wrapper that transforms its input as:
 * out = base_out + offset;
 */
template <typename Base>
class OffsetWrapper
{
public:

	typedef Base BaseType;
	typedef MatrixType OutputType;
	typedef typename BaseType::InputType InputType;

	OffsetWrapper( BaseType& b, const MatrixType& offset )
	: _base( b ), _offset( offset )
	{
		if( b.OutputSize().rows != offset.rows() ||
		    b.OutputSize().cols != offset.cols() )
		{
			throw std::runtime_error( "OffsetWrapper: Dimension mismatch." );
		}
	}

	unsigned int ParamDim() const { return _base.ParamDim(); }
	unsigned int OutputDim() const { return _base.OutputDim(); }
	MatrixSize OutputSize() const { return _base.OutputSize(); }
	
	unsigned int InputDim() const { return OutputDim(); }
	MatrixSize InputSize() const { return OutputSize(); }

	void SetParamsVec( const VectorType& v ) { _base.SetParamsVec( v ); }
	VectorType GetParamsVec() const { return _base.GetParamsVec(); }

	BackpropInfo Backprop( const InputType& input, const BackpropInfo& nextInfo )
	{
		return _base.Backprop( input, nextInfo );
	}

	OutputType Evaluate( const InputType& input ) const
	{
		return _base.Evaluate( input ) + _offset;
	}

private:

	BaseType& _base;
	MatrixType _offset;

};

}