#pragma once

#include "modprop/compo/Interfaces.h"
#include <iostream>

namespace percepto
{

class SquaredBoxLoss
: public Source<double>
{
public:

	typedef Source<VectorType> InputSourceType;
	typedef Sink<VectorType> SinkType;
	typedef Source<double> OutputSourceType;

	SquaredBoxLoss()
	: _input( this ), _weight( 1.0 ) {}

	SquaredBoxLoss( const SquaredBoxLoss& other )
	: _input( this ), _weight( other._weight ), 
	  _lower( other._lower ), 
	  _upper( other._upper ) {}

	void SetSource( InputSourceType* r ) { r->RegisterConsumer( &_input ); }

	void SetBounds( const VectorType& lower, const VectorType& upper )
	{
		if( lower.size() != upper.size() )
		{
			throw std::invalid_argument( "SquaredBoxLoss: Upper and lower bounds must be same size." );
		}
		_lower = lower;
		_upper = upper;
	}
	void SetWeight( double weight )
	{
		_weight = weight;
	}

	virtual unsigned int OutputDim() const { return 1; }

	virtual void Foreprop()
	{
		VectorType in = _input.GetInput();

		if( in.size() != _lower.size() )
		{
			throw std::invalid_argument( "SquaredBoxLoss: Input must be same size as bounds." );
		}

		VectorType lowerErr = in - _lower;
		VectorType upperErr = in - _upper;

		_violations = VectorType::Zero( in.size() );
		for( unsigned int i = 0; i < in.size(); ++i )
		{
			if( lowerErr(i) < 0 )
			{
				_violations(i) = lowerErr(i);
			}
			else if( upperErr(i) > 0 )
			{
				_violations(i) = upperErr(i);
			}
		}

		double err = 0.5 * _violations.dot( _violations ) * _weight;


		if( (_violations.array() != 0.0 ).any() )
		{
			// std::cout << "in: " << in.transpose() << std::endl;
			// std::cout << "lower: " << _lower.transpose() << std::endl;
			// std::cout << "upper: " << _upper.transpose() << std::endl;
			std::cout << "violations: " << _violations.transpose() << std::endl;
		}

		OutputSourceType::SetOutput( err );
		OutputSourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		if( nextDodx.size() == 0 )
		{
			_input.Backprop( _weight * _violations.transpose() );
		}
		else
		{
			_input.Backprop( nextDodx * _weight * _violations.transpose() );
		}
	}

private:

	SinkType _input;
	double _weight;
	VectorType _lower;
	VectorType _upper;
	VectorType _violations;

};

}