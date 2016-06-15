#pragma once
#include "percepto/PerceptoTypes.h"

namespace percepto
{

// Base class for all modules in the pipeline
class ModuleBase
{
public:

	ModuleBase() {}
	virtual ~ModuleBase() {}

	// Perform forward invalidation pass
	virtual void Invalidate() = 0;

	// Perform forward input propogation pass
	virtual void Foreprop() = 0;

	// Perform backward derivative propogation pass
	// If an empty nextDodx is given, this module is the terminal output
	virtual void Backprop( const MatrixType& nextDodx ) = 0;
};

template <typename Input>
class Sink
{
public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	typedef Input InputType;

	// Must be bound to owner that uses this sink
	Sink( ModuleBase* t ) 
	: _valid( false ), _owner( t ) {}
	
	virtual ~Sink() {}

	// Set the source for this input to Backprop along
	void SetSource( ModuleBase* src ) { _source = src; }

	void SetInput( const InputType& in ) 
	{ 
		_valid = true;
		_input = in;
		_owner->Foreprop();
	}

	virtual void UnsetInput()
	{
		// Don't notify if we're already invalid
		if( !_valid ) { return; }
		_owner->Invalidate();
		_valid = false;
	}

	bool IsValid() const { return _valid; }
	const InputType& GetInput() const { return _input; }

	virtual void Backprop( const MatrixType& nextDodx )
	{
		_source->Backprop( nextDodx );
	}

private:

	bool _valid;
	InputType _input; // TODO Make this a pointer instead to avoid copies
	ModuleBase* _owner;
	ModuleBase* _source;
};

// Needs Foreprop, Backprop to be defined
template <typename Output>
class Source
: public ModuleBase
{
public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	typedef Output OutputType;
	typedef Sink<OutputType> SinkType;

	Source() : _valid( false ) {}
	virtual ~Source() {}

	// Register a consumer of this source's output
	void RegisterConsumer( Sink<Output>* c ) 
	{ 
		_consumers.push_back( c ); 
		c->SetSource( this );
	}

	// Set this source's latched output
	virtual void SetOutput( const OutputType& o ) 
	{ 
		_output = o;
		_valid = true;
	}

	// Get this source's latched output
	const OutputType& GetOutput() const 
	{
		if( !_valid )
		{
			throw std::runtime_error( "Tried to get invalid source output. "
			                          + std::string("Did you forget to call Foreprop()?") );
		}
		return _output; 
	}

	virtual void Foreprop()
	{
		for( unsigned int i = 0; i < _consumers.size(); i++ )
		{
			_consumers[i]->SetInput( GetOutput() );
		}
	}

	// Invalidate all consumers of this source's output
	virtual void Invalidate()
	{
		for( unsigned int i = 0; i < _consumers.size(); i++ )
		{
			_consumers[i]->UnsetInput();
		}
		_valid = false;
	}

private:

	std::vector<SinkType*> _consumers;
	bool _valid;
	OutputType _output;
};

template <typename Output>
class TerminalSource
: public Source<Output>
{
public:

	typedef Output OutputType;
	typedef Source<Output> SourceType;

	TerminalSource() {}
	virtual ~TerminalSource() {}

	virtual void SetOutput( const OutputType& o )
	{
		_cache = o;
		SourceType::SetOutput( _cache );
	}

	virtual void Foreprop()
	{
		SourceType::SetOutput( _cache );
		SourceType::Foreprop();
	}

	virtual void Backprop( const MatrixType& nextDodx ) {}

private:

	OutputType _cache;
};

}