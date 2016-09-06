#pragma once

#include "modprop/ModpropTypes.h"

#include <iostream>

namespace percepto
{

struct AdamParameters
{
	// The step size (1E-3)
	double alpha;
	// The first moment memory term (0.9)
	double beta1;
	// The second moment memory term (0.999)
	double beta2;
	// The second moment norm offset (1E-7)
	double epsilon;
	// Whether or not to shrink the step size
	bool enableDecay;

	double maxStepElement;

	AdamParameters()
	: alpha( 1E-3 ), beta1( 0.9 ), beta2( 0.999 ), epsilon( 1E-7 ),
	  enableDecay( false ), maxStepElement( std::numeric_limits<double>::infinity() ) {}
};

// Based on work by Kingma and Ba. See (Kingma, Ba 2015)
class AdamStepper
{
public:

	/*! \brief Create a new stepper with the specified parameters. */
	AdamStepper( const AdamParameters& params = AdamParameters() )
	: _params( params )
	{
		if( _params.beta1 < 0 || _params.beta1 > 1.0 )
		{
			throw std::runtime_error( "beta1 must be between 0 and 1." );
		}
		if( _params.beta2 < 0 || _params.beta2 > 1.0 )
		{
			throw std::runtime_error( "beta2 must be between 0 and 1." );
		}
		Reset();
	}

	/*! \brief Reset the stepper state. */
	void Reset()
	{ 
		_t = 0; 
		_beta1Acc = _params.beta1;
		_beta2Acc = _params.beta2;
		_m = VectorType();
		// Don't need to reset _v since _m is checked
	}

	VectorType GetStep( const VectorType& gradient )
	{
		// Need to initialize state
		if( _m.size() == 0 )
		{
			_m = VectorType::Zero( gradient.size() );
			_v = VectorType::Zero( gradient.size() );
		}

		if( gradient.size() != _m.size() )
		{
			std::cout << "m.size(): " << _m.size() << std::endl;
			std::cout << "grad.size(): " << gradient.size() << std::endl;
			throw std::runtime_error( "Improperly sized gradient." );
		}

		++_t;
		_m = _params.beta1 * _m + (1.0 - _params.beta1 ) * gradient;
		VectorType gradientSq = ( gradient.array() * gradient.array() ).matrix();
		_v = _params.beta2 * _v + (1.0 - _params.beta2 ) * gradientSq;

		// VectorType mhat = _m / ( 1.0 - std::pow(_params.beta1, _t) );
		// VectorType vhat = _v / ( 1.0 - std::pow(_params.beta2, _t) );
		VectorType mhat = _m / ( 1.0 - _beta1Acc );
		VectorType vhat = _v / ( 1.0 - _beta2Acc );
		_beta1Acc *= _params.beta1;
		_beta2Acc *= _params.beta2;

		double step = _params.alpha;
		if( _params.enableDecay )
		{
			step = _params.alpha / std::sqrt( _t );
		}
		// std::cout << "step: " << step << std::endl;
		VectorType ret = ( step * mhat.array() / 
		         ( vhat.array().sqrt() + _params.epsilon ) ).matrix();

		double largestG = ret.array().abs().maxCoeff();
		if( largestG > _params.maxStepElement )
		{
			ret = ret / ( largestG / _params.maxStepElement );
		}
		return ret;
	}

private:

	AdamParameters _params;

	double _beta1Acc;
	double _beta2Acc;

	VectorType _m;
	VectorType _v;
	unsigned int _t;

};

}