#pragma once

#include "percepto/LinearLayer.hpp"

#include <sstream>

namespace percepto
{

// TODO Make this not necessarily a linear layer?
// TODO Include the activation parameters somehow so we can easily construct?
/*! \brief A fully-connected fixed-width network. */
template <typename Activation>
class FullyConnectedNet
{
public:

	typedef VectorType InputType;
	typedef VectorType OutputType;
	typedef LinearLayer<Activation> LayerType;
	typedef typename LayerType::ActivationType ActivationType;
	typedef std::vector< typename LayerType::ParamType> ParamType;

	// TODO Remove knowledge of LinearLayer internals
	static FullyConnectedNet<Activation>
	create_zeros( unsigned int inputDim, 
	              unsigned int outputDim,
	              unsigned int numHiddenLayers, 
	              unsigned int layerWidth,
	              const ActivationType& activation )
	{
		typename FullyConnectedNet<Activation>::ParamType params;
		params.reserve( numHiddenLayers );
		// Create input layer (first hidden layer)
		LayerType layer = LayerType::create_zeros( inputDim, layerWidth, activation );
		params.push_back( layer.GetParams() );
		
		// Create hidden layers
		layer = LayerType::create_zeros( layerWidth, layerWidth, activation );
		for( int i = 0; i < ((int)numHiddenLayers)-2; ++i )
		{
			params.push_back( layer.GetParams() );
		}

		// Create output layer (last hidden layer)
		layer = LayerType::create_zeros( layerWidth, outputDim, activation );
		params.push_back( layer.GetParams() );

		return FullyConnectedNet<Activation>( params, activation );
	}

	FullyConnectedNet( const ParamType& params, const ActivationType& activation )
	{
		layers.reserve( params.size() );
		for( unsigned int i = 0; i < params.size(); i++ )
		{
			layers.emplace_back( params[i], activation );
		}

		ValidateNet();
	}

	OutputType Evaluate( const InputType& input ) const
	{
		VectorType vec = input;
		for( unsigned int i = 0; i < layers.size(); ++i )
		{
			vec = layers[i].Evaluate( vec );
		}
		return vec;
	}

	BackpropInfo Backprop( const InputType& input,
	                      const BackpropInfo& nextNets ) const
	{
		unsigned int sysOutDim = nextNets.sysOutDim;
		BackpropInfo nextInfo = nextNets;
		BackpropInfo thisNets;
		thisNets.sysOutDim = sysOutDim;

		std::vector<InputType> inputs;
		inputs.reserve( layers.size() );
		inputs.push_back( input );
		for( unsigned int i = 0; i < layers.size(); ++i )
		{
			inputs.push_back( layers[i].Evaluate( inputs[i] ) );
		}

		thisNets.dodw = MatrixType::Zero( sysOutDim, ParamDim() );
		unsigned int paramIndex = ParamDim();

		for( int i = layers.size()-1; i >= 0; --i )
		{
			nextInfo = layers[i].Backprop( inputs[i], nextInfo );
			
			paramIndex = paramIndex - layers[i].ParamDim();
			thisNets.dodw.block( 0, paramIndex, sysOutDim, layers[i].ParamDim() ) 
			    = nextInfo.dodw;
		}
		thisNets.dodx = nextInfo.dodx;
		return thisNets;
	}

	void SetParams( const ParamType& params )
	{
		assert( params.size() == layers.size() );
		for( unsigned int i = 0; i < params.size(); ++i )
		{
			layers[i].SetParams( params[i] );
		}
	}

	void SetParamsVec( const VectorType& params )
	{
		assert( params.size() == ParamDim() );
		unsigned int vecInd = 0;
		for( unsigned int i = 0; i < layers.size(); ++i )
		{
			unsigned int d = layers[i].ParamDim();
			layers[i].SetParamsVec( params.block( vecInd, 0, d, 1 ) );
			vecInd += d;
		}
	}

	ParamType GetParams() const
	{
		ParamType params;
		params.reserve( layers.size() );
		for( unsigned int i = 0; i < layers.size(); ++i )
		{
			params.push_back( layers[i].GetParams() );
		}
		return params;
	}

	VectorType GetParamsVec() const
	{
		VectorType params( ParamDim() );
		unsigned int vecInd = 0;
		for( unsigned int i = 0; i < layers.size(); ++i )
		{
			unsigned int d = layers[i].ParamDim();
			params.block( vecInd, 0, d, 1 ) = layers[i].GetParamsVec();
			vecInd += d;
		}
		return params;
	}

	void StepParams( const VectorType& step )
	{
		assert( step.size() == ParamDim() );
		unsigned int paramInd = 0;
		for( unsigned int i = 0; i < layers.size(); ++i )
		{
			unsigned int dim = layers[i].ParamDim();
			layers[i].StepParams( step.block( paramInd, 0, dim, 1 ) );
			paramInd += dim;
		}
	}

	unsigned int InputDim() const { return layers.front().InputDim(); }
	unsigned int OutputDim() const { return layers.back().OutputDim(); }

	unsigned int ParamDim() const
	{
		unsigned int dim = 0;
		for( unsigned int i = 0; i < layers.size(); ++i )
		{
			dim += layers[i].ParamDim();
		}
		return dim;
	}

private:

	std::vector<LayerType> layers;

	void ValidateNet() const
	{
		for( unsigned int i = 1; i < layers.size(); i++ )
		{
			if( layers[i].InputDim() != layers[i-1].OutputDim() )
			{
				std::stringstream ss;
				ss << "Net dimension error: Layer " << i << " input dim is "
				   << layers[i].InputDim() << " but previous output dim is "
				   << layers[i-1].OutputDim();
				throw std::runtime_error( ss.str() );
			}
		}
	}

};

}
