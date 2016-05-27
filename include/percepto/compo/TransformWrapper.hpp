#pragma once
#include "percepto/compo/BackpropInfo.hpp"
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
	typedef typename BaseType::InputType InputType;

	TransformWrapper( BaseType& b, const MatrixType& transform )
	: _base( b ), _transform( transform )
	{
		if( transform.cols() != b.OutputSize().rows )
		{
			throw std::runtime_error( "TransformWrapper: Dimension mismatch." );
		}
	}

	unsigned int ParamDim() const { return _base.ParamDim(); }

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

	void SetParamsVec( const VectorType& v ) { _base.SetParamsVec( v ); }
	VectorType GetParamsVec() const { return _base.GetParamsVec(); }

	BackpropInfo Backprop( const InputType& input, const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		MatrixType dSdx( OutputDim(), InputDim() );
		MatrixType d = MatrixType::Zero( _base.OutputSize().rows, 
		                                 _base.OutputSize().cols );
		for( unsigned int i = 0; i < InputDim(); i++ )
		{
			// TODO Make more efficient with row/col product
			d(i) = 1;
			MatrixType temp = _transform * d * _transform.transpose();
			dSdx.col(i) = Eigen::Map<VectorType>( temp.data(), temp.size(), 1 );
			d(i) = 0;
		}

		BackpropInfo midInfo;
		midInfo.dodx = nextInfo.dodx * dSdx;
		return _base.Backprop( input, midInfo );
	}

	OutputType Evaluate( const InputType& input ) const
	{
		return _transform * _base.Evaluate( input ) *
		       _transform.transpose();
	}

private:

	BaseType& _base;
	MatrixType _transform;

};

}