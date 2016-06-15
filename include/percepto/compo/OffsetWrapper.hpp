#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/compo/BackpropInfo.hpp"
#include <memory>
#include <stdexcept>

namespace percepto
{

/** 
 * \brief A wrapper that transforms its input as:
 * out = base_out + offset;
 */
template <typename DataType>
class OffsetWrapper
: public Source<DataType>
{
public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	typedef Source<DataType> SourceType;
	typedef Sink<DataType> SinkType;
	typedef DataType OutputType;

	OffsetWrapper() 
	: _input( this ) {}

	OffsetWrapper( const OffsetWrapper& other )
	: _input( this ), _offset( other._offset ) {}

	void SetSource( SourceType* b ) { b->RegisterConsumer( &_input ); }
	void SetOffset( const DataType& offset ) { _offset = offset; }
	DataType GetOffset() const { return _offset; }

	virtual void Backprop( const MatrixType& nextDodx )
	{
		// std::cout << "OffsetWrapper backprop" << std::endl;
		_input.Backprop( nextDodx );
	}

	virtual void Foreprop()
	{
		SourceType::SetOutput( _input.GetInput() + _offset );
		SourceType::Foreprop();
	}

private:

	SinkType _input;
	OutputType _offset;
};

}