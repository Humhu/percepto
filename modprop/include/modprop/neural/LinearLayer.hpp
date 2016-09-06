#pragma once

#include "modprop/compo/Interfaces.h"
#include "modprop/compo/Parametric.hpp"
#include <iostream>

namespace percepto
{

/*! A fully-connected linear layer with a single shared
 * activation object. */
template <typename Activation>
class LinearLayer
: public Source<VectorType>
{
public:
	
	typedef Source<VectorType> SourceType;
	typedef VectorType InputType;
	typedef VectorType OutputType;
	typedef Activation ActivationType;

	static unsigned int compute_param_dim( unsigned int inputDim, 
	                                       unsigned int outputDim )
	{
		return outputDim * (inputDim + 1);
	}

	/*! Creates a layer with the specified dimensionality and all zero parameters. */
	LinearLayer( unsigned int inputDim, unsigned int outputDim,
	             const ActivationType& activation )
	: _inputDim( inputDim ), _outputDim( outputDim ), 
	_inputPort( this ), _params( nullptr ), _weights( nullptr, 0, 0 ),
	_activation( activation ) {}

	LinearLayer( const LinearLayer& other )
	: _inputDim( other._inputDim ), _outputDim( other._outputDim ),
	_inputPort( this ), _params( other._params ),
	_weights( other._weights ), _activation( other._activation ) {}

	Parameters::Ptr CreateParameters()
	{
		unsigned int dim = compute_param_dim( _inputDim, _outputDim );
		Parameters::Ptr params = std::make_shared<Parameters>();
		params->Initialize( dim );
		SetParameters( params );
		return params;
	}

	void SetOffsets( const VectorType& off )
	{
		RowMatrixType p( _weights );
		if( off.size() != p.rows() )
		{
			throw std::runtime_error( "LinearLayer: Offsets invalid dimension." );
		}
		p.col( p.cols() - 1 ) = off;
		Eigen::Map<VectorType> v( p.data(), p.size(), 1 );
		_params->SetParamsVec( v );
		new (&_weights) Eigen::Map<const RowMatrixType>( _params->GetParamsVec().data(),
		                                                 _outputDim, _inputDim + 1 );
	}

	void SetParameters( Parameters::Ptr params )
	{
		if( params->ParamDim() != compute_param_dim( _inputDim, _outputDim ) )
		{
			throw std::runtime_error( "LinearLayer: Invalid parameter dimension." );
		}
		_params = params;
		new (&_weights) Eigen::Map<const RowMatrixType>( params->GetParamsVec().data(),
		                                                 _outputDim, _inputDim + 1 );
	}

	void SetSource( SourceType* b ) 
	{ 
		b->RegisterConsumer( &_inputPort );
	}

	virtual void Foreprop()
	{
		const InputType& input = _inputPort.GetInput();

		OutputType out = _weights * input.homogeneous();
		for( unsigned int i = 0; i < OutputDim(); i++ )
		{
			out(i) = _activation.Evaluate( out(i) );
		}
		SourceType::SetOutput( out );
		SourceType::Foreprop();
	}

	// TODO Clean this up!
	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		const InputType& input = _inputPort.GetInput();
		
		MatrixType dody;
		if( nextDodx.size() == 0 )
		{
			dody = MatrixType::Identity( OutputDim(), OutputDim() );
		}
		else
		{
			dody = nextDodx;
		}

		// TODO Implement Forward/Backward semantics to avoid double evaluation
		OutputType preAct = _weights * input.homogeneous();

		MatrixType thisDodw = MatrixType::Zero( dody.rows(), ParamDim() );
		MatrixType thisDodx = MatrixType::Zero( dody.rows(), InputDim() );

		for( unsigned int j = 0; j < OutputDim(); j++ )
		{
			double actDeriv = _activation.Derivative( preAct(j) );
			for( unsigned int i = 0; i < dody.rows(); i++ )
			{
				thisDodw.block( i, j*(InputDim()+1), 1, InputDim()+1 )
				    = dody(i,j) * input.homogeneous().transpose() * actDeriv;
				thisDodx.row(i) += dody(i,j) * _weights.block(j, 0, 1, InputDim() ) * actDeriv;
			}
		}

		// std::cout << "input: " << input.transpose() << std::endl;
		// std::cout << "weights: " << std::endl << _weights << std::endl;
		// std::cout << "thisDodw: " << std::endl << thisDodw << std::endl;
		// std::cout << "thisDodx: " << std::endl << thisDodx << std::endl;

		_params->AccumulateDerivs( thisDodw );
		_inputPort.Backprop( thisDodx );
	}

	unsigned int InputDim() const { return _weights.cols() - 1; }
	MatrixSize OutputSize() const { return MatrixSize( OutputDim(), 1 ); }
	unsigned int OutputDim() const { return _weights.rows(); }
	virtual unsigned int ParamDim() const { return _weights.size(); }

	const ActivationType& GetActivation() const { return _activation; }

private:

	unsigned int _inputDim;
	unsigned int _outputDim;
	Sink<VectorType> _inputPort;
	Parameters::Ptr _params;
	Eigen::Map<const RowMatrixType> _weights;
	ActivationType _activation;

	template <typename A>
	friend std::ostream& operator<<( std::ostream& os, const LinearLayer<A>& l );

};

template <typename Activation>
std::ostream& operator<<( std::ostream& os, const LinearLayer<Activation>& l )
{
	os << l._weights;
	return os;
}

}