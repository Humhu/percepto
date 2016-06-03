#pragma once

#include "percepto/neural/LinearLayer.hpp"

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

	FullyConnectedNet( unsigned int inputDim, unsigned int outputDim,
	                   unsigned int numHiddenLayers, unsigned int layerWidth,
	                   const ActivationType& activation )
	{
		layers.reserve( numHiddenLayers + 1 );

		// First layer takes input to width
		layers.emplace_back( inputDim, layerWidth, activation );

		// Middle layers take width to width
		for( unsigned int i = 1; i < numHiddenLayers-1; ++i )
		{
			layers.emplace_back( layerWidth, layerWidth, activation );
		}

		// Last layer takes width to output
		layers.emplace_back( layerWidth, outputDim, activation );

		ValidateNet();
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
	                       const BackpropInfo& nextInfo ) const
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		BackpropInfo thisNets;

		// TODO Cache forward pass?
		std::vector<InputType> inputs;
		inputs.reserve( layers.size() );
		inputs.push_back( input );
		for( unsigned int i = 0; i < layers.size(); ++i )
		{
			inputs.push_back( layers[i].Evaluate( inputs[i] ) );
		}

		thisNets.dodw = MatrixType::Zero( nextInfo.SystemOutputDim(), ParamDim() );
		unsigned int paramIndex = ParamDim();

		BackpropInfo layerInfo = nextInfo;
		for( int i = layers.size()-1; i >= 0; --i )
		{
			layerInfo = layers[i].Backprop( inputs[i], layerInfo );
			
			paramIndex = paramIndex - layers[i].ParamDim();
			thisNets.dodw.block( 0, paramIndex, nextInfo.SystemOutputDim(), layers[i].ParamDim() ) 
			    = layerInfo.dodw;
		}
		thisNets.dodx = layerInfo.dodx;
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
