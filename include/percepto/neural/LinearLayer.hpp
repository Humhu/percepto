#pragma once

#include "percepto/compo/BackpropInfo.hpp"

namespace percepto
{

/*! A fully-connected linear layer with a single shared
 * activation object. */
template <typename Activation>
class LinearLayer
{
public:
	
	typedef Eigen::Matrix<double,-1,-1,Eigen::RowMajor> ParamType;
	typedef VectorType InputType;
	typedef VectorType OutputType;
	typedef Activation ActivationType;

	/*! Creates a layer with the specified dimensionality and all zero parameters. */
	static LinearLayer<Activation>
	create_zeros( unsigned int inputDim, unsigned int outputDim, 
	              const ActivationType& activation )
	{
		return LinearLayer<Activation>( MatrixType::Zero( outputDim, inputDim + 1 ),
		                                activation ); 
	}

	/*! Creates a layer with specified weight matrix and activation object.
	 * params - An output_dim x input_dim matrix
	 * activation - An activation object to copy and use for all outputs. */
	LinearLayer( const ParamType& params, const ActivationType& activation )
	: _weights( params ), _activation( activation ) {}

	OutputType Evaluate( const InputType& input ) const
	{
		assert( input.size() == InputDim() );
		OutputType out = _weights * input.homogeneous();
		for( unsigned int i = 0; i < OutputDim(); i++ )
		{
			out(i) = _activation.Evaluate( out(i) );
		}
		return out;
	}

	// TODO Clean this up!
	BackpropInfo Backprop( const InputType& input,
	                       const BackpropInfo& nextLayers ) const
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		BackpropInfo thisLayers;
		MatrixType dody = nextLayers.dodx;

		// TODO Implement Forward/Backward semantics to avoid double evaluation
		OutputType preAct = _weights * input.homogeneous();

		thisLayers.dodw = MatrixType::Zero( nextLayers.SystemOutputDim(), ParamDim() );
		thisLayers.dodx = MatrixType::Zero( nextLayers.SystemOutputDim(), InputDim() );
		for( unsigned int j = 0; j < OutputDim(); j++ )
		{
			double actDeriv = _activation.Derivative( preAct(j) );
			for( unsigned int i = 0; i < nextLayers.SystemOutputDim(); i++ )
			{
				thisLayers.dodw.block(i, j*(InputDim()+1), 1, InputDim()+1 )
				    = dody(i,j) * input.homogeneous().transpose() * actDeriv;
				thisLayers.dodx.row(i) += dody(i,j) * _weights.block(j, 0, 1, InputDim() ) * actDeriv;
			}
		}
		return thisLayers;
	}

	unsigned int InputDim() const { return _weights.cols() - 1; }
	unsigned int OutputDim() const { return _weights.rows(); }
	unsigned int ParamDim() const { return _weights.size(); }

	void SetParams( const ParamType& params )
	{
		assert( params.cols() == _weights.cols() &&
		        params.rows() == _weights.rows() );
		_weights = params;
	}

	void SetParamsVec( const VectorType& params )
	{
		assert( params.size() == _weights.size() );
		Eigen::Map<const ParamType> w( params.data(), _weights.rows(), _weights.cols() );
		_weights = w;
	}

	ParamType GetParams() const
	{
		return _weights;
	}

	ConstVectorViewType GetParamsVec() const
	{
		return Eigen::Map<const VectorType>( _weights.data(), _weights.size(), 1 );
	}

private:

	ParamType _weights;
	ActivationType _activation;

};

}