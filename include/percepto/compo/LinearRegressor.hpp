#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/compo/Parametric.hpp"
#include <Eigen/Dense>

namespace percepto 
{

// TODO Support matrix inputs?
/*! \brief A simple linear regression class. Outputs Weights matrix * input. 
 * The matrix and input are dynamically-sized. Assumes row-major ordering
 * for vectorizing the weights matrix. */
class LinearRegressor
: public Source<VectorType>
{
public:
	
	typedef VectorType InputType;
	typedef VectorType OutputType;
	typedef Source<VectorType> SourceType;

	LinearRegressor( unsigned int inputDim, unsigned int outputDim )
	: _inputDim( inputDim ), _outputDim( outputDim ), _W( nullptr, 0, 0 ),
	_inputPort( this ) {}

	LinearRegressor( const LinearRegressor& other )
	: _inputDim( other._inputDim ), _outputDim( other._outputDim ),
	_params( other._params ), _W( other._W ), _inputPort( this ) {}

	void SetSource( SourceType* b ) { b->RegisterConsumer( &_inputPort ); }

	unsigned int InputDim() const { return _inputDim; }
	unsigned int OutputDim() const { return _outputDim; }

	Parameters::Ptr CreateParameters()
	{
		Parameters::Ptr params = std::make_shared<Parameters>();
		params->Initialize( (_inputDim + 1) * _outputDim );
		SetParameters( params );
		return params;
	}

	void SetParameters( Parameters::Ptr params )
	{
		if( params->ParamDim() != (_inputDim+1) * _outputDim )
		{
			throw std::runtime_error( "LinearRegressor: Invalid parameter dimension." );
		}
		_params = params;
		new (&_W) Eigen::Map<const MatrixType>( params->GetParamsVec().data(),
		                                        _outputDim, _inputDim + 1 );
	}

	void SetOffsets( const VectorType& off )
	{
		MatrixType p( _W );
		p.col( p.cols() - 1 ) = off;
		Eigen::Map<VectorType> v( p.data(), p.size(), 1 );
		_params->SetParamsVec( v );
		new (&_W) Eigen::Map<const MatrixType>( _params->GetParamsVec().data(),
		                                              _outputDim, _inputDim + 1 );
	}

	virtual void Foreprop()
	{
		VectorType out = _W * _inputPort.GetInput().homogeneous();
		SourceType::SetOutput( out );
		SourceType::Foreprop();
	}
	
	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "LinearRegressor: Backprop dim error!" );
		}

		const VectorType& input = _inputPort.GetInput();
		MatrixType thisDodw = MatrixType( nextDodx.rows(), ParamDim() );
		for( unsigned int i = 0; i < nextDodx.rows(); i++ )
		{
			for( unsigned int j = 0; j < OutputDim(); j++ )
			{
				thisDodw.block(i, j*(InputDim()+1), 1, InputDim()+1) =
					nextDodx(i,j) * input.homogeneous().transpose();
			}
		}
		MatrixType thisDodx = nextDodx * _W;

		_params->AccumulateDerivs( thisDodw );
		_inputPort.Backprop( thisDodx );
	}

	virtual unsigned int ParamDim() const { return _W.size(); }

private:

	friend std::ostream& operator<<( std::ostream& os, const LinearRegressor& lreg );
	
	unsigned int _inputDim;
	unsigned int _outputDim;
	Parameters::Ptr _params;
	Eigen::Map<const MatrixType> _W; // The weight vector
	Sink<VectorType> _inputPort;
	
};

inline
std::ostream& operator<<( std::ostream& os, const LinearRegressor& lreg )
{
	os << lreg._W;
	return os;
}

}