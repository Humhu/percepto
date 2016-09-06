#pragma once

#include "modprop/compo/Interfaces.h"

#define PMF_NORM_TOL (1E-6)

namespace percepto
{

// Represents the log-probability of selecting a particular action
// from a discrete probability distribution
class DiscreteLogProbability
: public Source<double>
{
public:

	typedef Source<double> OutputSourceType;
	typedef Source<VectorType> InputSourceType;
	typedef Sink<VectorType> SinkType;

	DiscreteLogProbability()
	: _input( this ) {}

	DiscreteLogProbability( const DiscreteLogProbability& other )
	: _input( this ), _index( other._index ) {}

	void SetSource( InputSourceType* s )
	{
		s->RegisterConsumer( &_input );
	}

	void SetIndex( unsigned int ind )
	{
		_index = ind;
	}

	virtual void Foreprop()
	{
		VectorType pmf = _input.GetInput();
		if( std::abs( pmf.sum() - 1.0 ) > PMF_NORM_TOL )
		{
			throw std::runtime_error( "DiscreteLogProbability: PMF not normalized." );
		}
		if( _index >= pmf.size() )
		{
			throw std::runtime_error( "DiscreteLogProbability: Index out of range." );
		}

		double lp_i = std::log( pmf(_index) );
		OutputSourceType::SetOutput( lp_i );
		OutputSourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		VectorType pmf = _input.GetInput();
		double p_i = pmf( _index );
		
		// Maybe make this epsilon of some sort
		if( p_i == 0.0 )
		{
			throw std::runtime_error( "DiscreteLogProbability: Zero probability indexed." );
		}

		if( nextDodx.cols() != 1 )
		{
			throw std::runtime_error( "DiscreteLogProbability: nextDodx dimension error." );
		}

		MatrixType dydx = MatrixType::Constant( 1, pmf.size(), -1.0 );
		dydx( 0, _index ) += 1.0/p_i;

		if( nextDodx.size() == 0 )
		{
			_input.Backprop( dydx );
		}
		else
		{
			_input.Backprop( nextDodx * dydx );
		}
	}

private:

	SinkType _input;
	unsigned int _index;
};

}

