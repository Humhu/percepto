#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/PerceptoTypes.h"
#include <iostream>

namespace percepto
{

template <typename DataType>
class Repeater
: public Source<Output>,
public Sink<Output>
{
public:

	typedef Source<Output> SourceType;
	typedef Sink<Output> SinkType;

	Repeater()
	: SourceType(), SinkType( this ) {}

	virtual void Foreprop()
	{
		SourceType::SetOutput( SinkType::GetInput() );
	}

	virtual void Backprop()
	{

	}

private:


};

template <typename Input, typename Output>
class InputWrapper
: public Source<Output>
{
public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	typedef Input InputType;
	typedef Output OutputType;
	typedef Source<Output> SourceType;
	typedef Sink<Input> SinkType;

	PipelineWrapper() 
	: _output( nullptr ) {}

	void RegisterTerminalSource( ModuleBase* src )
	{
		_sources.push_back( src );
	}

	void SetOutputSource( SourceType* dst )
	{
		_output = dst;
	}

	virtual void Invalidate()
	{
		for( unsigned int i = 0; i < _sources.size(); i++ )
		{
			_sources[i]->Invalidate();
		}
		SourceType::Invalidate();
	}

	virtual void Foreprop()
	{
		Invalidate();
		for( unsigned int i = 0; i < _sources.size(); i++ )
		{
			_sources[i].Foreprop();
		}
	}

private:

	std::vector<ModuleBase*> _sources;
	SourceType* _dst;
};

}