#pragma once
#include "percepto/PerceptoTypes.h"
#include <memory>
#include <stdexcept>

namespace percepto
{

/** 
 * \brief A wrapper that transforms its input as:
 * out = transform * base_out * transform.transpose()
 */
template <typename Base>
class TransformWrapper
{
public:

	typedef Base BaseType;
	typedef MatrixType OutputType;

	TransformWrapper( BaseType& b, const MatrixType& transform )
	: _base( b ), _transform( transform )
	{
		if( transform.cols() != b.OutputSize().rows )
		{
			throw std::runtime_error( "TransformWrapper: Dimension mismatch." );
		}
	}

	unsigned int OutputDim() const { return _transform.rows() * _transform.rows(); }
	MatrixSize OutputSize() const 
	{ 
		return MatrixSize( _transform.rows(), _transform.rows() ); 
	}
	
	unsigned int InputDim() const { return _transform.cols() * _transform.cols(); }
	MatrixSize InputSize() const
	{
		return MatrixSize( _transform.cols(), _transform.cols() );
	}

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "TransformWrapper: Backprop dim error." );
		}

		MatrixType dSdx( OutputDim(), InputDim() );
		MatrixType d = MatrixType::Zero( _base.OutputSize().rows, 
		                                 _base.OutputSize().cols );
		for( unsigned int i = 0; i < InputDim(); i++ )
		{
			d(i) = 1;
			MatrixType temp = _transform * d * _transform.transpose();
			dSdx.col(i) = Eigen::Map<VectorType>( temp.data(), temp.size(), 1 );
			d(i) = 0;
		}

		MatrixType thisDodx = nextDodx * dSdx;
		_base.Backprop( thisDodx );
		return thisDodx;
	}

	OutputType Evaluate() const
	{
		return _transform * _base.Evaluate() *
		       _transform.transpose();
	}

private:

	BaseType& _base;
	MatrixType _transform;

};

}