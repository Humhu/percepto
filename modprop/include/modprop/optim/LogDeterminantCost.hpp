#pragma once

#include "modprop/compo/Interfaces.h"

#include <Eigen/Cholesky>
#include <iostream>

namespace percepto
{

// TODO Support other types of decompositions
template <typename Data>
class LogDeterminantCost
: public Source<double>
{
public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	typedef Data InputType;
	typedef Source<Data> InputSourceType;
	typedef Sink<Data> SinkType;
	typedef Source<double> OutputSourceType;

	LogDeterminantCost()
	: _input( this ), _scale( 1.0 ), _initialized( false ) {}

	LogDeterminantCost( const LogDeterminantCost& other )
	: _input( this ), _scale( other._scale ), _initialized( other._initialized ),
	_dody( other.dody ) {}

	void SetSource( InputSourceType* r ) { r->RegisterConsumer( &_input ); }
	void SetScale( double s ) { _scale = s; }

	virtual unsigned int OutputDim() const { return 1; }

	virtual void Foreprop()
	{
		InputType input = _input.GetInput();
		if( input.rows() != input.cols() )
		{
			throw std::invalid_argument( "LogDeterminantCost: Input must be square." );
		}

		_initialized = false;

		_solver = _solver.compute( input );
		VectorType d( _solver.vectorD() );
		if( !( d.array() > 0.0 ).all() )
		{
			std::cout << "in: " << input << std::endl;
			std::cout << "D: " << d.transpose() << std::endl;
			throw std::invalid_argument( "LogDeterminantCost: Input determinant must be positive." );
		}
		double logdet = 0;
		for( unsigned int i = 0; i < _solver.vectorD().size(); ++i )
		{
			logdet += std::log( _solver.vectorD()(i) );;
		}
		OutputSourceType::SetOutput( logdet );
		OutputSourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		if( !_initialized )
		{
			InputType input = _input.GetInput();
			InputType inputInv = _solver.solve( MatrixType::Identity( input.rows(), input.cols() ) );
			std::cout << "input: " << input << std::endl;
			std::cout << "inputInv: " << inputInv << std::endl;
			_dody = Eigen::Map<MatrixType>( inputInv.data(), 1, inputInv.size() );
			_initialized = true;
		}

		if( nextDodx.size() == 0 )
		{
			_input.Backprop( _dody );
		}
		else
		{
			std::cout << "thisDodx: " << nextDodx * _dody << std::endl;
			_input.Backprop( nextDodx * _dody );
		}
	}

private:

	typedef Eigen::LDLT<Data> SolverType;
	
	SinkType _input;
	double _scale;
	bool _initialized;
	MatrixType _dody;
	SolverType _solver;
};

}