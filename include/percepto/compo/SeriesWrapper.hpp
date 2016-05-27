#pragma once

#include "percepto/utils/MatrixUtils.hpp"
#include "percepto/compo/BackpropInfo.hpp"

namespace percepto
{

// TODO Generalize to N series elements?
// Joins two networks together into a single network object by
// taking the output of head and feeding it into tail
template <typename Head, typename Tail>
class SeriesWrapper
{
public:

	typedef Head HeadType;
	typedef typename HeadType::InputType InputType;
	typedef Tail TailType;
	typedef typename TailType::OutputType OutputType;

	SeriesWrapper( HeadType& head, TailType& tail )
	: _head( head ), _tail( tail ) 
	{
		assert( head.OutputDim() == tail.InputDim() );
	}

	OutputType Evaluate( const InputType& input ) const
	{
		return _tail.Evaluate( _head.Evaluate( input ) );
	}

	BackpropInfo Backprop( const InputType& input,
	                       const BackpropInfo& nextNets ) const
	{
		BackpropInfo tailInfo = _tail.Backprop( _head.Evaluate( input ), nextNets );
		BackpropInfo headInfo = _head.Backprop( input, tailInfo );
		BackpropInfo thisInfo;
		
		thisInfo.dodw = ConcatenateHor( headInfo.dodw, tailInfo.dodw );
		thisInfo.dodx = headInfo.dodx * tailInfo.dodx;
		return thisInfo;
	}

	unsigned int InputDim() const { return _head.InputDim(); }
	unsigned int OutputDim() const { return _tail.OutputDim(); }
	unsigned int ParamDim() const { return _head.ParamDim() + _tail.ParamDim(); }

	void SetParamsVec( const VectorType& params )
	{
		_head.SetParamsVec( params.head( _head.ParamDim() ) );
		_tail.SetParamsVec( params.tail( _tail.ParamDim() ) );
	}

	VectorType GetParamsVec() const
	{
		VectorType vec( ParamDim() );
		vec << _head.GetParamsVec(), _tail.GetParamsVec();
		return vec;
	}

private:

	HeadType& _head;
	TailType& _tail;

};

}