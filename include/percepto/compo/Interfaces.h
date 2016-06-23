#pragma once

#include "percepto/PerceptoTypes.h"
#include <iostream>

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
		// if( !nextDodx.allFinite() )
		// {
		// 	throw std::runtime_error( "Non-finite backprop" );
		// }
		_source->Backprop( nextDodx );
	}

private:

	Sink( const Sink& other );
	Sink& operator=( const Sink& other );

	bool _valid;
	InputType _input; // TODO Make this a pointer instead to avoid copies
	ModuleBase* _owner;
	ModuleBase* _source;
};

// Needs Backprop to be overridden
template <typename Output>
class Source
: public ModuleBase
{
public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	typedef Output OutputType;
	typedef Sink<OutputType> SinkType;

	std::string name;

	Source() : _valid( false ), _backpropsReceived( 0 ) {}
	virtual ~Source() {}

	// Register a consumer of this source's output
	void RegisterConsumer( Sink<Output>* c ) 
	{
		_consumers.push_back( c ); 
		c->SetSource( this );
	}

	bool IsValid() const { return _valid; }

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
			                          + std::string( "Did you forget to call Foreprop()?" ) );
		}
		return _output; 
	}

	virtual void Backprop( const MatrixType& nextDodx ) final
	{
		if( _dodxAcc.size() == 0 )
		{
			_dodxAcc = nextDodx;
		}
		else
		{
			_dodxAcc += nextDodx;
		}
		_backpropsReceived++;

		// if( !name.empty() )
		// {
		// 	std::cout << name << " has " << std::to_string( _backpropsReceived )
		// 	          << " out of " << _consumers.size() << std::endl;
		// }
		
		// If we are the terminal source (get an empty nextDodx), then 
		// we will have 0 consumers. We need to then propagate immediately
		// so this check looks for >= than # consumers
		if( _backpropsReceived >= _consumers.size() )
		{
			// if( !name.empty() )
			// {
			// 	std::cout << "Backpropping: " << name << std::endl;
			// }
			BackpropImplementation( _dodxAcc );
		}
	}

	// Should be implemented by derived class to be called when all backprops
	// are accumualted and ready to be further backpropped
	virtual void BackpropImplementation( const MatrixType& nextDodx ) = 0;

	virtual void Foreprop()
	{
		for( unsigned int i = 0; i < _consumers.size(); i++ )
		{
			_consumers[i]->SetInput( GetOutput() );
		}
	}

	// Invalidate all consumers of this source's output and reset
	// all backprop accumulators
	virtual void Invalidate()
	{
		_valid = false;
		_backpropsReceived = 0;
		_dodxAcc = MatrixType();
		for( unsigned int i = 0; i < _consumers.size(); i++ )
		{
			_consumers[i]->UnsetInput();
		}
	}

private:

	Source( const Source& other );
	Source& operator=( const Source& other );

	std::vector<SinkType*> _consumers;
	bool _valid;
	OutputType _output;

	unsigned int _backpropsReceived;
	MatrixType _dodxAcc;
};

template <typename Output>
class TerminalSource
: public Source<Output>
{
public:

	typedef Output OutputType;
	typedef Source<Output> SourceType;
	std::string name;

	TerminalSource() : name("") {}

	TerminalSource( const std::string& n ) : name( n ) {}

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

	virtual void BackpropImplementation( const MatrixType& nextDodx ) 
	{
		// std::cout << "Terminal source backprop: " << name << std::endl;
	}

private:

	OutputType _cache;
};

}