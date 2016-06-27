#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/PerceptoTypes.h"
#include <memory>
#include <stdexcept>

namespace percepto
{

class VectorTransformWrapper
: public Source<VectorType>
{
public:

	typedef Source<VectorType> SourceType;
	typedef Sink<VectorType> SinkType;
	typedef VectorType OutputType;

	VectorTransformWrapper()
	: _input( this ) {}

	VectorTransformWrapper( const VectorTransformWrapper& other )
	: _input( this ) {}

	void SetSource( SourceType* b ) { b->RegisterConsumer( &_input ); }
	void SetTransform( const MatrixType& m ) { _transform = m; }

	virtual void Foreprop()
	{
		SourceType::SetOutput( _transform * _input.GetInput() );
		SourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != _transform.rows() )
		{
			std::cout << "nextDodx: " << nextDodx << std::endl;
			std::cout << "_transform.rows(): " << _transform.rows() << std::endl;
			throw std::runtime_error( "VectorTransformWrapper: Backprop dim error." );
		}
		_input.Backprop( nextDodx * _transform );
	}
private:

	SinkType _input;
	MatrixType _transform;
};

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

	TransformWrapper( const TransformWrapper& other ) 
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
	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != _transform.rows() * _transform.rows() )
		{
			throw std::runtime_error( "TransformWrapper: Backprop dim error!" );
		}
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