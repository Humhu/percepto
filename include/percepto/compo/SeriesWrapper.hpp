#pragma once

#include "percepto/utils/MatrixUtils.hpp"
#include "percepto/compo/BackpropInfo.hpp"
#include <deque>
#include <iostream>

namespace percepto
{

// Joins two networks together into a single network object by
// taking the output of head and feeding it into tail
template <typename Head, typename Tail>
class SeriesWrapper
{
public:

	typedef Head HeadType;
	typedef Tail TailType;
	typedef typename TailType::OutputType OutputType;

	SeriesWrapper( HeadType& head, TailType& tail )
	: _head( head ), _tail( tail ) 
	{}

	OutputType Evaluate() const
	{
		return _tail.Evaluate( _head.Evaluate() );
	}

	MatrixType Backprop( const MatrixType& nextDodx ) const
	{
		MatrixType tailDodx = _tail.Backprop( _head.Evaluate(), nextDodx );
		return _head.Backprop( tailDodx );
	}

	MatrixSize OutputSize() const { return _tail.OutputSize(); }
	unsigned int OutputDim() const { return _tail.OutputDim(); }

private:

	HeadType& _head;
	TailType& _tail;
};

template <typename Head, typename Tail, 
          template<typename,typename> class Container = std::deque>
class SequenceWrapper
{
public:

	typedef Head HeadType;
	typedef Tail TailType;
	typedef Container<TailType, std::allocator<TailType>> ContainerType;
	typedef typename Tail::OutputType OutputType;

	SequenceWrapper( HeadType& head, ContainerType& tails ) 
	: _head( head ), _tails( tails ) {}

	OutputType Evaluate() const
	{
		OutputType out = _head.Evaluate();
		for( unsigned int i = 0; i < _tails.size(); i++ )
		{
			out = _tails[i].Evaluate( out );
		}
		return out;
	}

	MatrixSize OutputSize() const { return _tails.back().OutputSize(); }
	unsigned int OutputDim() const { return _tails.back().OutputDim(); }

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "SequenceWrapper: Backprop dim error." );
		}

		std::vector<OutputType> inputs;
		inputs.reserve( _tails.size() + 1 );
		inputs.push_back( _head.Evaluate() );
		for( unsigned int i = 0; i < _tails.size(); ++i )
		{
			inputs.push_back( _tails[i].Evaluate( inputs[i] ) );
		}

		MatrixType layerDodx = nextDodx;
		for( int i = _tails.size()-1; i >= 0; --i )
		{
			layerDodx = _tails[i].Backprop( inputs[i], layerDodx );
		}
		// This module's input is head's input

		return _head.Backprop( layerDodx );
	}

private:

	HeadType& _head;
	ContainerType& _tails;
};

}