#pragma once

#include "percepto/compo/Interfaces.h"

namespace percepto
{

// Wrapper to perform forward/backward passes
// It consists of terminal sources where outputs originate, and
// one terminal sink where all outputs terminate
class Pipeline
{
public:

	Pipeline() 
	: _sink( nullptr ) {}

	// NOTE Could check for duplicates but it isn't fatal
	void RegisterTerminalSource( ModuleBase* src ) 
	{ 
		_sources.push_back( src ); 
	}

	void SetTerminalSink( ModuleBase* dst )
	{
		_sink = dst;
	}

	void Invalidate()
	{
		for( unsigned int i = 0; i < _sources.size(); i++ )
		{
			_sources[i]->Invalidate();
		}
	}

	void Foreprop() 
	{
		for( unsigned int i = 0; i < _sources.size(); i++ )
		{
			_sources[i]->Foreprop();
		}
	}

	void Backprop()
	{
		_sink->Backprop( MatrixType() );
	}

private:

	std::vector<ModuleBase*> _sources;
	ModuleBase* _sink;
};

}