#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/PerceptoTypes.h"
#include <memory>
#include <stdexcept>
#include <iostream>
namespace percepto
{

/** 
 * \brief A wrapper that transforms its input as:
 * out = transform * base_out * transform.transpose()
 */
class TransformWrapper
: public Source<MatrixType>
{
public:

	typedef Source<MatrixType> SourceType;
	typedef Sink<MatrixType> SinkType;
	typedef MatrixType OutputType;

	TransformWrapper() 
	: _input( this ) {}

	void SetSource( SourceType* b ) { b->RegisterConsumer( &_input ); }
	void SetTransform( const MatrixType& transform ) { _transform = transform; }

	virtual void Foreprop()
	{
		SourceType::SetOutput( _transform * _input.GetInput() *
		                       _transform.transpose() );
		SourceType::Foreprop();
	}

	// TODO Check for empty nextDodx
	virtual void Backprop( const MatrixType& nextDodx )
	{
		const MatrixType& input = _input.GetInput();
		unsigned int inDim = _transform.cols() * _transform.cols();
		unsigned int outDim = _transform.rows() * _transform.rows();
		MatrixType dSdx( outDim, inDim );
		MatrixType d = MatrixType::Zero( input.rows(), 
		                                 input.cols() );
		for( unsigned int i = 0; i < inDim; i++ )
		{
			d(i) = 1;
			MatrixType temp = _transform * d * _transform.transpose();
			dSdx.col(i) = Eigen::Map<VectorType>( temp.data(), temp.size(), 1 );
			d(i) = 0;
		}

		if( nextDodx.size() == 0 )
		{
			_input.Backprop( dSdx );
		}
		else
		{
			_input.Backprop( nextDodx * dSdx );
		}
	}

private:

	SinkType _input;
	MatrixType _transform;

};

}