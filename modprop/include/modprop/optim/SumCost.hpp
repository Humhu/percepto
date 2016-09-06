#pragma once

#include <boost/foreach.hpp>
#include <deque>
#include "modprop/compo/Interfaces.h"

namespace percepto
{

template <typename DataType>
class SumCost
: public Source<DataType>
{
public:

	typedef Source<DataType> SourceType;
	typedef Sink<DataType> SinkType;
	typedef DataType OutputType;

	SumCost() {}

	void AddSource( SourceType* s )
	{
		_sinks.emplace_back( this );
		s->RegisterConsumer( &_sinks.back() );
	}

	// TODO Come up with a better way to do this
	// Using a list allows remove, but makes random sampling hard
	void RemoveOldestSource()
	{
		if( _sinks.size() == 0 ) { return; }
		_sinks.pop_front();
	}

	virtual void Foreprop()
	{
		if( _sinks.size() == 0 ) { return; }
		BOOST_FOREACH( const SinkType& input, _sinks )
		{
			if( !input.IsValid() ) { return; }
		}

		OutputType out = _sinks[0].GetInput();
		for( unsigned int i = 1; i < _sinks.size(); ++i )
		{
			out += _sinks[i].GetInput();
		}
		SourceType::SetOutput( out );
		SourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		if( _sinks.size() == 0 ) { return; }
		BOOST_FOREACH( SinkType& input, _sinks )
		{
			input.Backprop( nextDodx );
		}
	}

	// TODO Take a comparator as a template parameter
	OutputType ComputeMax() const
	{
		OutputType largest = -std::numeric_limits<OutputType>::infinity();
		BOOST_FOREACH( const SinkType& input, _sinks )
		{
			OutputType out = input.GetInput();
			if( out > largest ) { largest = out; }
		}
		return largest;
	}

protected:

	// Using a deque so that growing does not invalidate pointers and
	// we can efficiently remove items
	std::deque<SinkType> _sinks;

private:

	// Disallow copying b/c sinks can't copy correctly
	SumCost( const SumCost& other );

};

}