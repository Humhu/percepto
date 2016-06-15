#pragma once

#include <boost/foreach.hpp>
#include <deque>
#include "percepto/compo/Interfaces.h"

namespace percepto
{

template <typename DataType>
class MeanCost
: public Source<DataType>
{
public:

	typedef Source<DataType> SourceType;
	typedef Sink<DataType> SinkType;
	typedef DataType OutputType;

	MeanCost() {}

	void AddSource( SourceType* s )
	{
		_sinks.emplace_back( this );
		s->RegisterConsumer( &_sinks.back() );
	}

	virtual void Foreprop()
	{
		BOOST_FOREACH( const SinkType& input, _sinks )
		{
			if( !input.IsValid() ) { return; }
		}

		OutputType out = _sinks[0].GetInput();
		for( unsigned int i = 1; i < _sinks.size(); ++i )
		{
			out += _sinks[i].GetInput();
		}
		SourceType::SetOutput( out / _sinks.size() );
		SourceType::Foreprop();
	}

	virtual void Backprop( const MatrixType& nextDodx )
	{
		MatrixType thisDodx = nextDodx / _sinks.size();
		BOOST_FOREACH( SinkType& input, _sinks )
		{
			input.Backprop( thisDodx );
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

	// Using a deque so that growing does not invalidate pointers
	std::deque<SinkType> _sinks;

};

}