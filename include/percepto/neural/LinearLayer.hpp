#pragma once

#include "percepto/compo/Parametric.hpp"
#include <iostream>

namespace percepto
{

/*! A fully-connected linear layer with a single shared
 * activation object. */
template <typename Activation>
class LinearLayer
: public Parametric
{
public:
	
	typedef Eigen::Matrix<double,-1,-1,Eigen::RowMajor> ParamType;
	typedef VectorType InputType;
	typedef VectorType OutputType;
	typedef Activation ActivationType;

	/*! Creates a layer with the specified dimensionality and all zero parameters. */
	LinearLayer( unsigned int inputDim, unsigned int outputDim,
	             const ActivationType& activation )
	: _weights( ParamType::Zero( outputDim, inputDim + 1 ) ),
	_activation( activation ) {}

	/*! Creates a layer with specified weight matrix and activation object.
	 * params - An output_dim x input_dim matrix
	 * activation - An activation object to copy and use for all outputs. */
	LinearLayer( const ParamType& params, const ActivationType& activation )
	: _weights( params ), _activation( activation ) {}

	OutputType Evaluate( const InputType& input ) const
	{
		if( input.size() != InputDim() )
		{
			std::cout << "input size: " << input.size() << " dim: " << InputDim() << std::endl;
			throw std::runtime_error( "LinearLayer: Input dim mismatch." );
		}

		OutputType out = _weights * input.homogeneous();
		for( unsigned int i = 0; i < OutputDim(); i++ )
		{
			out(i) = _activation.Evaluate( out(i) );
		}
		return out;
	}

	// TODO Clean this up!
	MatrixType Backprop( const InputType& input,
	                     const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "LinearLayer: Backprop dim error." );
		}

		MatrixType dody = nextDodx;

		// TODO Implement Forward/Backward semantics to avoid double evaluation
		OutputType preAct = _weights * input.homogeneous();

		MatrixType thisDodw = MatrixType::Zero( nextDodx.rows(), ParamDim() );
		MatrixType thisDodx = MatrixType::Zero( nextDodx.rows(), InputDim() );
		for( unsigned int j = 0; j < OutputDim(); j++ )
		{
			double actDeriv = _activation.Derivative( preAct(j) );
			for( unsigned int i = 0; i < nextDodx.rows(); i++ )
			{
				thisDodw.block(i, j*(InputDim()+1), 1, InputDim()+1 )
				    = dody(i,j) * input.homogeneous().transpose() * actDeriv;
				thisDodx.row(i) += dody(i,j) * _weights.block(j, 0, 1, InputDim() ) * actDeriv;
			}
		}

		Parametric::AccumulateWeightDerivs( thisDodw );
		return thisDodx;
	}

	unsigned int InputDim() const { return _weights.cols() - 1; }
	MatrixSize OutputSize() const { return MatrixSize( OutputDim(), 1 ); }
	unsigned int OutputDim() const { return _weights.rows(); }
	virtual unsigned int ParamDim() const { return _weights.size(); }

	void SetParams( const ParamType& params )
	{
		assert( params.cols() == _weights.cols() &&
		        params.rows() == _weights.rows() );
		_weights = params;
	}

	virtual void SetParamsVec( const VectorType& params )
	{
		assert( params.size() == _weights.size() );
		Eigen::Map<const ParamType> w( params.data(), _weights.rows(), _weights.cols() );
		_weights = w;
	}

	ParamType GetParams() const
	{
		return _weights;
	}

	virtual VectorType GetParamsVec() const
	{
		return Eigen::Map<const VectorType>( _weights.data(), _weights.size(), 1 );
	}

	const ActivationType& GetActivation() const { return _activation; }

private:

	ParamType _weights;
	ActivationType _activation;

};

}