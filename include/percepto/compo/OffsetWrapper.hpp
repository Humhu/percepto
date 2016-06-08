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

	OffsetWrapper( BaseType& b, const MatrixType& offset )
	: _base( b ), _offset( offset )
	{
		if( b.OutputSize().rows != offset.rows() ||
		    b.OutputSize().cols != offset.cols() )
		{
			throw std::runtime_error( "OffsetWrapper: Dimension mismatch." );
		}
	}

	unsigned int OutputDim() const { return _base.OutputDim(); }
	MatrixSize OutputSize() const { return _base.OutputSize(); }
	
	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "OffsetWrapper: Backprop dim error." );
		}
		
		_base.Backprop( nextDodx );
		return nextDodx;
	}

	OutputType Evaluate() const
	{
		return _base.Evaluate() + _offset;
	}

private:

	BaseType& _base;
	MatrixType _offset;

};

}